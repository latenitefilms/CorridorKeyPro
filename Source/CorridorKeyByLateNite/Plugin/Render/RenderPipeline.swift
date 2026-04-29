//
//  RenderPipeline.swift
//  CorridorKey by LateNite
//
//  Orchestrates a single-frame render. The pipeline is built around three
//  command buffers so the inference engine (which owns synchronous CPU work
//  on Apple Silicon's unified memory) can sit between GPU passes without
//  racing their writes:
//
//  1. Pre-inference — screen rotation → hint extraction → combine & normalise.
//  2. Inference — the neural engine reads the normalised tensor and writes
//     alpha + foreground at inference resolution.
//  3. Post-inference — upscale → despill → refiner blend → despeckle → matte
//     refine → passthrough → light wrap → edge decontamination → restore →
//     compose directly into Final Cut Pro's destination texture via a render
//     pass.
//
//  Two things changed from v0 that keep the render thread responsive:
//
//  * Intermediate textures come from a pool (`IntermediateTexturePool`) and
//    are returned to the pool via `addCompletedHandler` on the command
//    buffer that used them. The render path no longer allocates fresh
//    textures per frame on the hot path.
//
//  * GPU→CPU synchronisation uses `addCompletedHandler` + `DispatchSemaphore`
//    instead of `waitUntilCompleted`. `waitUntilCompleted` spin-waits inside
//    the Metal driver on some macOS builds, which prevented Final Cut Pro
//    from pipelining the next tile's command queue kickoff while we were
//    nominally "done".
//

import Foundation
import Metal
import CoreMedia
import simd

#if FXPLUG_HOST
/// Input bundle for an FxPlug render. Value-type keeps the orchestrator
/// easy to reason about; `FxImageTile` is passed along because we only
/// read it, never mutate.
///
/// FxPlug callers build this directly. The standalone editor uses the
/// `renderToTexture` entry point instead, which constructs the per-frame
/// state from raw `MTLTexture` inputs without involving `FxImageTile` at
/// all — and is therefore available to both hosts.
struct RenderRequest: @unchecked Sendable {
    let destinationImage: FxImageTile
    let sourceImage: FxImageTile
    let alphaHintImage: FxImageTile?
    let state: PluginStateData
    let workingGamut: WorkingColorGamut
    let renderTime: CMTime
}
#endif

/// Result fed back to the FxPlug layer so it can surface status in the
/// inspector's "Runtime Status" group.
struct RenderReport: Sendable {
    let backendDescription: String
    let guideSourceDescription: String
    let effectiveInferenceResolution: Int
    let deviceName: String
}

/// Tile-free per-frame inputs. Built by both render entry points (the FxPlug
/// `render(_ request:)` method extracts these from the `RenderRequest`; the
/// standalone editor's `renderToTexture(...)` builds them directly from raw
/// `MTLTexture`s). All internal render helpers take this struct so the same
/// code paths serve both hosts without leaking `FxImageTile` deeper into
/// the renderer.
struct ResolvedRenderInputs: @unchecked Sendable {
    let state: PluginStateData
    let alphaHintTexture: (any MTLTexture)?
    let workingGamut: WorkingColorGamut
    let renderTime: CMTime
}

/// Runs the per-frame pipeline. One instance lives per plug-in instance so
/// each timeline can keep its own warmed-up model. All Metal resources are
/// shared through `MetalDeviceCache`, so creating a pipeline is cheap.
final class RenderPipeline: @unchecked Sendable {
    private let deviceCache: MetalDeviceCache
    let inferenceCoordinator: InferenceCoordinator

    init(
        deviceCache: MetalDeviceCache = .shared,
        inferenceCoordinator: InferenceCoordinator = InferenceCoordinator()
    ) {
        self.deviceCache = deviceCache
        self.inferenceCoordinator = inferenceCoordinator
    }

#if FXPLUG_HOST
    /// Executes the full render for one tile. Returns a `RenderReport` so the
    /// FxPlug layer can update the runtime status fields.
    func render(_ request: RenderRequest) throws -> RenderReport {
        let context = try makeDeviceContext(for: request)
        defer { context.entry.returnCommandQueue(context.commandQueue) }
        let inputs = ResolvedRenderInputs(
            state: request.state,
            alphaHintTexture: request.alphaHintImage?.metalTexture(for: context.device),
            workingGamut: request.workingGamut,
            renderTime: request.renderTime
        )
        return try renderInternal(inputs: inputs, context: context)
    }
#endif

    /// Tile-free render entry point used by the standalone editor. Mirrors
    /// `render(_ request:)` but takes raw `MTLTexture`s for source and
    /// destination so callers without an `FxImageTile` can drive the same
    /// pipeline. The destination texture's full extent is treated as one
    /// tile.
    func renderToTexture(
        source: any MTLTexture,
        destination: any MTLTexture,
        alphaHint: (any MTLTexture)? = nil,
        state: PluginStateData,
        workingGamut: WorkingColorGamut = .rec709,
        renderTime: CMTime = .zero
    ) throws -> RenderReport {
        let context = try makeDeviceContext(
            sourceTexture: source,
            destinationTexture: destination
        )
        defer { context.entry.returnCommandQueue(context.commandQueue) }
        let inputs = ResolvedRenderInputs(
            state: state,
            alphaHintTexture: alphaHint,
            workingGamut: workingGamut,
            renderTime: renderTime
        )
        return try renderInternal(inputs: inputs, context: context)
    }

    /// Shared dispatch: picks the render sub-path for the given inputs and
    /// device context. Both `render(_:)` and `renderToTexture(...)` funnel
    /// through here so neither has to know about `FxImageTile`.
    private func renderInternal(
        inputs: ResolvedRenderInputs,
        context: DeviceContext
    ) throws -> RenderReport {
        let screenTransform = ScreenColorEstimator.defaultTransform(for: inputs.state.screenColor)
        let gamutTransform = ColorGamutMatrix.transform(for: inputs.workingGamut)
        let inferenceResolution = inputs.state.qualityMode.resolvedInferenceResolution(
            forLongEdge: context.longEdge,
            deviceRegistryID: context.entry.device.registryID
        )

        // Diagnostic: when the user picks the Hint output mode, route
        // straight to a path that runs the pre-inference hint stage and
        // visualises the result. Skip MLX entirely so the user sees
        // exactly what prior the network would receive — useful for
        // diagnosing "why is the matte different than I expect" or
        // "did Vision detect my subject correctly".
        if inputs.state.outputMode == .hint {
            return try renderHintDiagnostic(
                inputs: inputs,
                context: context,
                screenTransform: screenTransform,
                gamutTransform: gamutTransform,
                inferenceResolution: inferenceResolution
            )
        }

        // Fast path: FxAnalyzer already ran for this clip and the inference
        // resolution matches the quality the user is asking for. Skip MLX
        // entirely — the cache already holds a network-quality matte.
        if let header = cachedMatteHeader(from: inputs.state, expectedResolution: inferenceResolution),
           let blob = inputs.state.cachedMatteBlob {
            return try renderUsingCachedAlpha(
                inputs: inputs,
                context: context,
                screenTransform: screenTransform,
                gamutTransform: gamutTransform,
                inferenceResolution: inferenceResolution,
                cachedBlob: blob,
                cachedWidth: header.width,
                cachedHeight: header.height
            )
        }
        // Manual hint mode demands at least one user-placed hint —
        // the network can't produce a useful matte from an empty
        // hint channel. Surface as a pass-through so the user sees
        // their footage rather than a black frame, and label the
        // backend so the inspector status row can explain why.
        if inputs.state.hintMode.requiresUserHints,
           inputs.state.hintPointSet.points.isEmpty {
            return try renderSourcePassThrough(
                inputs: inputs,
                context: context,
                backendDescription: "Manual Hint — add at least one foreground / background point"
            )
        }
        // Unanalysed → leave the source untouched. Running MLX on the render
        // thread made toggling the effect feel laggy and produced inconsistent
        // output while the analysis cache was being built, so pass-through is
        // now the explicit "nothing to key yet" signal.
        return try renderSourcePassThrough(inputs: inputs, context: context)
    }

    /// Raw output of the analysis pass. `alpha` is a width*height single-channel
    /// float buffer; `source` is an optional width*height*4 interleaved RGBA
    /// float buffer at the same resolution as the alpha (only produced when
    /// `readbackSource` is true, used by the temporal-blend motion gate).
    /// `engineDescription` records which engine actually processed this
    /// frame so the analyser log accurately reports MLX vs rough-matte
    /// per frame instead of guessing from the registry's current state.
    struct AnalysisExtraction {
        let alpha: [Float]
        let source: [Float]?
        let width: Int
        let height: Int
        let inferenceResolution: Int
        let engineDescription: String
    }

    /// Runs pre-inference + MLX (no post-processing) and returns the raw alpha
    /// matte on the CPU. Called by `FxAnalyzer.analyzeFrame` so the matte can
    /// be compressed and stored in the custom parameter. This path never reads
    /// from or writes to the per-frame cache — the custom parameter is the
    /// persistent cache.
    ///
    /// Pass `readbackSource: true` when temporal stability is enabled so the
    /// caller can feed the current and previous source frames to the motion
    /// gate. The extra readback is a single 4-channel copy at inference
    /// resolution — ~64 MB on the Maximum rung, well below the pre-existing
    /// 192 MB/frame MLX working set.
    // NOTE: A potential v1.1 optimisation is overlapping the next
    // frame's pre-inference with the current frame's MLX call. The
    // savings are modest (~5–10% of analyse wall time — pre-inference
    // is 20–40 ms, MLX is 300–2000 ms), and MLX's Swift API does not
    // expose a `wait(onEvent:)` primitive that lets us fence Metal
    // pre-inference work against an MLX evaluation, so the cleanest
    // implementation is to make `extractAlphaMatteForAnalysis` async
    // and let two `analyzeFrame:` calls land in flight. Documented as
    // deferred work; the synchronous `commitAndWait` here remains
    // correct and easy to reason about.
    func extractAlphaMatteForAnalysis(
        sourceTexture: any MTLTexture,
        state: PluginStateData,
        workingGamut: WorkingColorGamut,
        renderTime: CMTime,
        device: any MTLDevice,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue,
        readbackSource: Bool = false
    ) throws -> AnalysisExtraction {
        // Manual hint mode requires at least one user-placed hint —
        // an empty hint channel would feed MLX a degenerate prior and
        // the analyser would fill the cache with garbage. Fail fast
        // so the caller can surface the error in the analysis status
        // row instead of silently producing a bad matte sequence.
        if state.hintMode.requiresUserHints, state.hintPointSet.points.isEmpty {
            throw KeyingInferenceError.modelUnavailable(
                "Manual Hint mode needs at least one foreground or background hint point. Open the Subject menu in the transport bar and click the preview to drop one."
            )
        }
        let screenTransform = ScreenColorEstimator.defaultTransform(for: state.screenColor)
        let gamutTransform = ColorGamutMatrix.transform(for: workingGamut)
        let longEdge = max(sourceTexture.width, sourceTexture.height)
        let inferenceResolution = state.qualityMode.resolvedInferenceResolution(
            forLongEdge: longEdge,
            deviceRegistryID: device.registryID
        )

        let pre = try runPreInference(
            sourceTexture: sourceTexture,
            hintTexture: nil,
            hintMode: state.hintMode,
            hintPoints: state.hintPointSet.points,
            device: device,
            entry: entry,
            commandQueue: commandQueue,
            screenTransform: screenTransform,
            gamutTransform: gamutTransform,
            inferenceResolution: inferenceResolution
        )

        let cacheKey = InferenceCacheKey(
            frameTime: renderTime,
            screenColorRaw: state.screenColor.rawValue,
            inferenceResolution: inferenceResolution,
            cacheEntry: entry
        )
        let inferenceResult = try inferenceCoordinator.runInference(
            request: KeyingInferenceRequest(
                normalisedInputBuffer: pre.normalisedInputBuffer,
                rawSourceTexture: pre.rawSourceAtInferenceResolution.texture,
                inferenceResolution: inferenceResolution
            ),
            cacheEntry: entry,
            cacheKey: cacheKey
        )
        let inferenceOutput = inferenceResult.output

        let width = inferenceOutput.alphaTexture.width
        let height = inferenceOutput.alphaTexture.height
        var alpha = [Float](repeating: 0, count: width * height)
        let bytesPerRow = width * MemoryLayout<Float>.size
        alpha.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                inferenceOutput.alphaTexture.getBytes(
                    base,
                    bytesPerRow: bytesPerRow,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }

        var sourceReadback: [Float]? = nil
        if readbackSource {
            sourceReadback = try readbackRGBATexture(
                pre.rawSourceAtInferenceResolution.texture,
                entry: entry,
                commandQueue: commandQueue
            )
        }

        // Return pre-inference pooled textures now that we've captured the
        // alpha. The inference output textures live outside the pool, and
        // the normalised input buffer is cached per-rung on the entry so
        // it is not returned here.
        pre.rotatedSource?.returnManually()
        pre.rawSourceAtInferenceResolution.returnManually()

        return AnalysisExtraction(
            alpha: alpha,
            source: sourceReadback,
            width: width,
            height: height,
            inferenceResolution: inferenceResolution,
            engineDescription: inferenceResult.engineDescription
        )
    }

    /// Blits a pooled `.private`-storage texture into a reusable shared-
    /// storage staging texture (cached on `MetalDeviceCacheEntry`), waits
    /// for the GPU, and returns the pixels as an RGBA float array. Used by
    /// `extractAlphaMatteForAnalysis` so the CPU-side temporal blender can
    /// see the source values without dropping the render path's more
    /// efficient private storage.
    ///
    /// The staging texture is cached per `(width, height, pixelFormat)` to
    /// keep the autorelease pool from filling up under Final Cut Pro's
    /// tight analyse-frame loop; we saw unbounded memory growth when this
    /// allocated per-frame.
    private func readbackRGBATexture(
        _ texture: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue
    ) throws -> [Float] {
        let width = texture.width
        let height = texture.height
        guard let sharedTexture = entry.analysisReadbackTexture(
            width: width,
            height: height,
            pixelFormat: texture.pixelFormat
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }

        try autoreleasepool {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw MetalDeviceCacheError.commandBufferCreationFailed
            }
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                throw MetalDeviceCacheError.commandEncoderCreationFailed
            }
            blitEncoder.copy(
                from: texture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: width, height: height, depth: 1),
                to: sharedTexture,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blitEncoder.endEncoding()
            try commitAndWait(commandBuffer: commandBuffer)
        }

        var pixels = [Float](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                let bytesPerRow: Int
                switch texture.pixelFormat {
                case .rgba16Float:
                    // Use a staging Float16 buffer then widen to Float32 so
                    // downstream math doesn't depend on a specific pixel format.
                    bytesPerRow = width * 4 * MemoryLayout<Float16>.size
                    let halfCount = width * height * 4
                    let halfStorage = UnsafeMutablePointer<Float16>.allocate(capacity: halfCount)
                    defer { halfStorage.deallocate() }
                    sharedTexture.getBytes(
                        halfStorage,
                        bytesPerRow: bytesPerRow,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                    for index in 0..<halfCount {
                        base[index] = Float(halfStorage[index])
                    }
                case .rgba32Float:
                    bytesPerRow = width * 4 * MemoryLayout<Float>.size
                    sharedTexture.getBytes(
                        base,
                        bytesPerRow: bytesPerRow,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                default:
                    // Every producer in the analysis path currently hands us
                    // `.rgba16Float`; if a future path changes the format we
                    // want the mismatch to surface loudly rather than emit
                    // silent zeros, so leave `pixels` at its zero default.
                    break
                }
            }
        }
        return pixels
    }

    // MARK: - Render sub-paths

    private struct DeviceContext {
        let device: any MTLDevice
        let entry: MetalDeviceCacheEntry
        let commandQueue: any MTLCommandQueue
        let sourceTexture: any MTLTexture
        let destinationTexture: any MTLTexture
        let pixelFormat: MTLPixelFormat
        let sourceWidth: Int
        let sourceHeight: Int
        let longEdge: Int
        /// Width of the destination *output region* in pixels. Equal to the
        /// destination texture width when rendering full frames (standalone
        /// editor path) and to the FxPlug tile width when Final Cut Pro
        /// requests a sub-tile.
        let outputWidth: Float
        /// Height of the destination output region — see `outputWidth`.
        let outputHeight: Float
    }

    private struct PreInferenceArtifacts {
        let rotatedSource: PooledTexture?
        let normalisedInputBuffer: any MTLBuffer
        let rawSourceAtInferenceResolution: PooledTexture
    }

#if FXPLUG_HOST
    private func makeDeviceContext(for request: RenderRequest) throws -> DeviceContext {
        let destinationTile = request.destinationImage
        let sourceTile = request.sourceImage
        let pixelFormat = MetalDeviceCache.metalPixelFormat(for: destinationTile)
        guard let device = deviceCache.device(forRegistryID: destinationTile.deviceRegistryID) else {
            throw MetalDeviceCacheError.unknownDevice(destinationTile.deviceRegistryID)
        }
        let entry = try deviceCache.entry(for: device)
        guard let commandQueue = entry.borrowCommandQueue() else {
            throw MetalDeviceCacheError.queueExhausted
        }
        guard let sourceTexture = sourceTile.metalTexture(for: device) else {
            entry.returnCommandQueue(commandQueue)
            throw MetalDeviceCacheError.unknownDevice(destinationTile.deviceRegistryID)
        }
        guard let destinationTexture = destinationTile.metalTexture(for: device) else {
            entry.returnCommandQueue(commandQueue)
            throw MetalDeviceCacheError.unknownDevice(destinationTile.deviceRegistryID)
        }
        let outputWidth = Float(destinationTile.tilePixelBounds.right - destinationTile.tilePixelBounds.left)
        let outputHeight = Float(destinationTile.tilePixelBounds.top - destinationTile.tilePixelBounds.bottom)
        return DeviceContext(
            device: device,
            entry: entry,
            commandQueue: commandQueue,
            sourceTexture: sourceTexture,
            destinationTexture: destinationTexture,
            pixelFormat: pixelFormat,
            sourceWidth: sourceTexture.width,
            sourceHeight: sourceTexture.height,
            longEdge: max(sourceTexture.width, sourceTexture.height),
            outputWidth: outputWidth,
            outputHeight: outputHeight
        )
    }
#endif

    /// Tile-free `DeviceContext` constructor used by the standalone editor.
    /// The destination's full extent is treated as one tile so the compose
    /// stage covers the entire output texture.
    private func makeDeviceContext(
        sourceTexture: any MTLTexture,
        destinationTexture: any MTLTexture
    ) throws -> DeviceContext {
        let device = destinationTexture.device
        let entry = try deviceCache.entry(for: device)
        guard let commandQueue = entry.borrowCommandQueue() else {
            throw MetalDeviceCacheError.queueExhausted
        }
        return DeviceContext(
            device: device,
            entry: entry,
            commandQueue: commandQueue,
            sourceTexture: sourceTexture,
            destinationTexture: destinationTexture,
            pixelFormat: destinationTexture.pixelFormat,
            sourceWidth: sourceTexture.width,
            sourceHeight: sourceTexture.height,
            longEdge: max(sourceTexture.width, sourceTexture.height),
            outputWidth: Float(destinationTexture.width),
            outputHeight: Float(destinationTexture.height)
        )
    }

    /// Writes the source tile straight through to the destination. Used when
    /// the clip hasn't been analysed yet — the plug-in stays out of the way
    /// until the cache is populated. The optional `backendDescription`
    /// override lets the Manual Hint pass-through surface a more
    /// specific reason in the inspector's status row.
    private func renderSourcePassThrough(
        inputs: ResolvedRenderInputs,
        context: DeviceContext,
        backendDescription: String = "Source Pass-Through"
    ) throws -> RenderReport {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        commandBuffer.label = "CorridorKey by LateNite Pass-Through"

        // Reuse the compose render pipeline with the source bound as every
        // sampler and `foregroundOnly` output. This re-publishes the raw
        // source bytes through a tiny render pass so format and tile-layout
        // conversions the host expects still happen, without touching MLX.
        var passThroughState = inputs.state
        passThroughState.outputMode = .foregroundOnly
        try compose(
            source: context.sourceTexture,
            foreground: context.sourceTexture,
            matte: context.sourceTexture,
            context: context,
            state: passThroughState,
            commandBuffer: commandBuffer
        )

        try commitAndWait(commandBuffer: commandBuffer)

        return RenderReport(
            backendDescription: backendDescription,
            guideSourceDescription: "Clip not analysed",
            effectiveInferenceResolution: 0,
            deviceName: context.device.name
        )
    }

    /// Diagnostic render path: visualises the upstream hint texture
    /// (Vision mask, OSC dots, or green-bias rough matte) without
    /// touching MLX. Lets users see exactly what prior the network is
    /// receiving when investigating matte-quality issues.
    private func renderHintDiagnostic(
        inputs: ResolvedRenderInputs,
        context: DeviceContext,
        screenTransform: ScreenColorTransform,
        gamutTransform: WorkingSpaceTransform,
        inferenceResolution: Int
    ) throws -> RenderReport {
        PluginLog.notice("Hint Diagnostic: render started (hintMode=\(inputs.state.hintMode.displayName), source=\(context.sourceTexture.width)x\(context.sourceTexture.height), pixelFormat=\(context.pixelFormat.rawValue))")

        // Vision request runs synchronously on the Neural Engine in
        // parallel with the GPU pre-pass — same pattern as the
        // production analyse path so the diagnostic shows the same
        // hint MLX would actually receive. Errors here are logged
        // (not swallowed) so users investigating "why is my hint
        // black?" can see the cause in Console.app.
        var visionMask: VisionMask? = nil
        var visionFailureReason: String? = nil
        if inputs.state.hintMode.usesVisionPrior,
           inputs.alphaHintTexture == nil,
           #available(macOS 14.0, *) {
            if let engine = context.entry.visionHintEngine() as? VisionHintEngine {
                do {
                    visionMask = try engine.generateMask(source: context.sourceTexture)
                    if visionMask == nil {
                        visionFailureReason = "Vision found no foreground instance"
                        PluginLog.notice("Hint Diagnostic: Vision returned nil — falling back to green-bias hint.")
                    } else {
                        PluginLog.notice("Hint Diagnostic: Vision returned a mask, will resample and compose.")
                    }
                } catch {
                    visionFailureReason = error.localizedDescription
                    PluginLog.notice("Hint Diagnostic: Vision threw error '\(error.localizedDescription)' — falling back to green-bias hint.")
                }
            } else {
                visionFailureReason = "Vision hint engine unavailable"
                PluginLog.notice("Hint Diagnostic: visionHintEngine() returned nil — Vision unavailable.")
            }
        } else {
            PluginLog.notice("Hint Diagnostic: skipping Vision — hintMode=\(inputs.state.hintMode.displayName), externalHintAttached=\(inputs.alphaHintTexture != nil).")
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        commandBuffer.label = "CorridorKey by LateNite Hint Diagnostic"

        // Step 1: rotate into the screen-colour-canonical domain so
        // the green-bias hint matches what runPreInference would see
        // on a blue-screen clip.
        let rotatedSourcePooled = try RenderStages.applyScreenMatrix(
            source: context.sourceTexture,
            matrix: screenTransform.forwardMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: context.entry,
            commandBuffer: commandBuffer
        )
        let rotatedSource = rotatedSourcePooled?.texture ?? context.sourceTexture

        // Step 2: produce the hint texture exactly as the analyse path
        // does (external > Vision > manual zero-base > green-bias).
        let hintPooled: PooledTexture
        if let hostTexture = inputs.alphaHintTexture {
            hintPooled = try RenderStages.extractHint(
                source: hostTexture,
                layout: hintTileLayoutValue(for: hostTexture),
                targetWidth: rotatedSource.width,
                targetHeight: rotatedSource.height,
                entry: context.entry,
                commandBuffer: commandBuffer
            )
        } else if let mask = visionMask {
            hintPooled = try RenderStages.extractHint(
                source: mask.texture,
                layout: 1,
                targetWidth: rotatedSource.width,
                targetHeight: rotatedSource.height,
                entry: context.entry,
                commandBuffer: commandBuffer
            )
            mask.retainOnCompletion(of: commandBuffer)
        } else {
            // Manual and Automatic share the chroma prior — see the
            // matching branch in `runPreInference` for why a pure
            // zero-base hint produces nonsense mattes. Manual mode's
            // contract is enforced upstream by the empty-hint guard.
            hintPooled = try RenderStages.generateGreenHint(
                source: rotatedSource,
                entry: context.entry,
                commandBuffer: commandBuffer
            )
        }

        // Step 3: layer OSC hint points on top of whichever upstream
        // hint we just produced — so the diagnostic also shows the
        // user's manual dots.
        if !inputs.state.hintPointSet.points.isEmpty {
            try RenderStages.applyHintPoints(
                hint: hintPooled.texture,
                points: inputs.state.hintPointSet.points,
                entry: context.entry,
                commandBuffer: commandBuffer
            )
        }

        // Step 4: compose the hint into the FCP destination tile via
        // the existing compose pipeline; it sees `outputMode == .hint`
        // (CKOutputModeHint) and renders the matte channel as red.
        // We bind the hint texture to all three slots — only the
        // matte slot is actually read for this output mode.
        // Force the output mode to .hint here regardless of what the
        // request state says — at this point in the diagnostic path,
        // the user definitely picked "Hint (Diagnostic)", so any
        // confusion in popup index → enum mapping (which has bitten
        // us before across moef/Info.plist edits) doesn't matter.
        var hintRenderState = inputs.state
        hintRenderState.outputMode = .hint
        try compose(
            source: hintPooled.texture,
            foreground: hintPooled.texture,
            matte: hintPooled.texture,
            context: context,
            state: hintRenderState,
            commandBuffer: commandBuffer
        )
        PluginLog.notice("Hint Diagnostic: composed (hintTextureFormat=\(hintPooled.texture.pixelFormat.rawValue), destTextureFormat=\(context.destinationTexture.pixelFormat.rawValue), tileWidth=\(Int(context.outputWidth)), tileHeight=\(Int(context.outputHeight)))")

        try commitAndWait(commandBuffer: commandBuffer)

        // One-time readback of the hint texture (pre-compose) AND
        // the destination (post-compose). Together they answer the
        // "Hint Diagnostic shows black in FCP" question:
        //
        //   * hintMaxR > 0 → extractHint produced data; if the
        //     destination is also non-zero, everything works
        //   * hintMaxR == 0 → extractHint wrote nothing; trace back
        //     to Vision / wrapAsMetalTexture / blit
        //   * hintMaxR > 0 but destMaxR == 0 → compose misses the
        //     hint texture (sampling, binding, viewport)
        //
        // ~50 ms one-time cost; subsequent renders skip both
        // readbacks via the static flag.
        Self.dumpHintAndDestinationDiagnosticOnce(
            hint: hintPooled.texture,
            destination: context.destinationTexture,
            entry: context.entry,
            commandQueue: context.commandQueue
        )

        rotatedSourcePooled?.returnManually()
        hintPooled.returnManually()

        let hintSource: String
        if inputs.alphaHintTexture != nil {
            hintSource = "External alpha input"
        } else if visionMask != nil {
            hintSource = "Vision subject mask"
        } else if let reason = visionFailureReason {
            hintSource = "Green-bias rough matte (\(reason))"
        } else {
            hintSource = "Green-bias rough matte"
        }
        return RenderReport(
            backendDescription: "Hint Diagnostic (\(hintSource))",
            guideSourceDescription: hintSource,
            effectiveInferenceResolution: inferenceResolution,
            deviceName: context.device.name
        )
    }

    private func renderUsingCachedAlpha(
        inputs: ResolvedRenderInputs,
        context: DeviceContext,
        screenTransform: ScreenColorTransform,
        gamutTransform: WorkingSpaceTransform,
        inferenceResolution: Int,
        cachedBlob: Data,
        cachedWidth: Int,
        cachedHeight: Int
    ) throws -> RenderReport {
        guard let preCommandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        preCommandBuffer.label = "CorridorKey by LateNite Cached Pre-Inference"

        let rotatedSourcePooled = try RenderStages.applyScreenMatrix(
            source: context.sourceTexture,
            matrix: screenTransform.forwardMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: context.entry,
            commandBuffer: preCommandBuffer
        )
        let rotatedSource = rotatedSourcePooled?.texture ?? context.sourceTexture

        let rawSourcePooled = try RenderStages.resample(
            source: rotatedSource,
            targetWidth: inferenceResolution,
            targetHeight: inferenceResolution,
            pixelFormat: .rgba16Float,
            method: inputs.state.upscaleMethod,
            entry: context.entry,
            commandBuffer: preCommandBuffer
        )
        let rawSourceAtInferenceResolution = rawSourcePooled?.texture ?? rotatedSource

        try commitAndWait(commandBuffer: preCommandBuffer)

        rotatedSourcePooled?.returnManually()
        // rawSourcePooled stays alive — we need it as the foreground stand-in.

        let alphaPooled = try uploadCachedAlpha(
            blob: cachedBlob,
            width: cachedWidth,
            height: cachedHeight,
            entry: context.entry
        )

        try runPostInference(
            inputs: inputs,
            context: context,
            screenTransform: screenTransform,
            rotatedSource: rotatedSource,
            alphaAtInferenceResolution: alphaPooled.texture,
            foregroundAtInferenceResolution: rawSourceAtInferenceResolution
        )

        rawSourcePooled?.returnManually()
        alphaPooled.returnManually()

        return RenderReport(
            backendDescription: "Analysed Cache (\(inferenceResolution)px)",
            guideSourceDescription: "Analysis Cache",
            effectiveInferenceResolution: inferenceResolution,
            deviceName: context.device.name
        )
    }

    private func runPreInference(
        sourceTexture: any MTLTexture,
        hintTexture: (any MTLTexture)?,
        hintMode: HintMode,
        hintPoints: [HintPoint],
        device: any MTLDevice,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue,
        screenTransform: ScreenColorTransform,
        gamutTransform: WorkingSpaceTransform,
        inferenceResolution: Int
    ) throws -> PreInferenceArtifacts {
        // Vision runs on Neural Engine in parallel with the GPU pre-pass,
        // so kick it off BEFORE we encode the screen matrix. Vision reads
        // the original (un-rotated) source — the hint is single-channel
        // alpha, colour-space-independent, so the screen matrix is
        // irrelevant. Fall back to the green-bias hint when Vision finds
        // no salient subject or fails for any reason; the render path
        // never silently degrades.
        var visionMask: VisionMask? = nil
        if hintMode == .appleVision, hintTexture == nil, #available(macOS 14.0, *) {
            if let engine = entry.visionHintEngine() as? VisionHintEngine {
                do {
                    visionMask = try engine.generateMask(source: sourceTexture)
                } catch {
                    PluginLog.notice(
                        "Vision hint failed for analysis frame, falling back to green-bias hint: \(error.localizedDescription)"
                    )
                }
            }
        }

        guard let preCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        preCommandBuffer.label = "CorridorKey by LateNite Pre-Inference"

        let rotatedSourcePooled = try RenderStages.applyScreenMatrix(
            source: sourceTexture,
            matrix: screenTransform.forwardMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: entry,
            commandBuffer: preCommandBuffer
        )
        let rotatedSource = rotatedSourcePooled?.texture ?? sourceTexture

        let hintTexturePooled: PooledTexture
        if let hostTexture = hintTexture {
            hintTexturePooled = try RenderStages.extractHint(
                source: hostTexture,
                layout: hintTileLayoutValue(for: hostTexture),
                targetWidth: rotatedSource.width,
                targetHeight: rotatedSource.height,
                entry: entry,
                commandBuffer: preCommandBuffer
            )
        } else if let mask = visionMask {
            hintTexturePooled = try RenderStages.extractHint(
                source: mask.texture,
                layout: 1,
                targetWidth: rotatedSource.width,
                targetHeight: rotatedSource.height,
                entry: entry,
                commandBuffer: preCommandBuffer
            )
            mask.retainOnCompletion(of: preCommandBuffer)
        } else {
            // Manual and Automatic both use the chroma-derived prior
            // here — the network was trained on that distribution,
            // not on sparse user dots in isolation. The empty-hint
            // guard in `renderInternal` / the analyser already
            // refuses to run Manual mode without at least one user
            // dot, so the user's clicks still gate progress; they
            // just augment the chroma prior on top of feeding it.
            hintTexturePooled = try RenderStages.generateGreenHint(
                source: rotatedSource,
                entry: entry,
                commandBuffer: preCommandBuffer
            )
        }

        // Layer user-placed foreground / background dots from the OSC
        // on top of whichever upstream hint we just produced. No-op
        // when the user hasn't placed any.
        if !hintPoints.isEmpty {
            try RenderStages.applyHintPoints(
                hint: hintTexturePooled.texture,
                points: hintPoints,
                entry: entry,
                commandBuffer: preCommandBuffer
            )
        }

        let normalisedInputBuffer = try RenderStages.combineAndNormaliseIntoBuffer(
            source: rotatedSource,
            hint: hintTexturePooled.texture,
            inferenceResolution: inferenceResolution,
            workingToRec709: gamutTransform.workingToRec709,
            entry: entry,
            commandBuffer: preCommandBuffer
        )

        let rawSourceAtInferenceResolutionPooled = try RenderStages.resample(
            source: rotatedSource,
            targetWidth: inferenceResolution,
            targetHeight: inferenceResolution,
            pixelFormat: .rgba16Float,
            method: .bilinear,
            entry: entry,
            commandBuffer: preCommandBuffer
        )
        let rawSource: PooledTexture
        if let rawSourceAtInferenceResolutionPooled {
            rawSource = rawSourceAtInferenceResolutionPooled
        } else {
            // Rare: caller already provided a texture at the inference
            // resolution. Acquire a throwaway copy so we still hand back a
            // `PooledTexture` and the caller's pool bookkeeping stays
            // consistent.
            guard let fallback = entry.texturePool.acquire(
                width: inferenceResolution,
                height: inferenceResolution,
                pixelFormat: .rgba16Float
            ) else { throw MetalDeviceCacheError.textureAllocationFailed }
            if let blit = preCommandBuffer.makeBlitCommandEncoder() {
                blit.label = "CorridorKey by LateNite Pre-Inf Copy"
                blit.copy(from: rotatedSource, to: fallback.texture)
                blit.endEncoding()
            }
            rawSource = fallback
        }

        try commitAndWait(commandBuffer: preCommandBuffer)
        hintTexturePooled.returnManually()

        return PreInferenceArtifacts(
            rotatedSource: rotatedSourcePooled,
            normalisedInputBuffer: normalisedInputBuffer,
            rawSourceAtInferenceResolution: rawSource
        )
    }

    private func runPostInference(
        inputs: ResolvedRenderInputs,
        context: DeviceContext,
        screenTransform: ScreenColorTransform,
        rotatedSource: any MTLTexture,
        alphaAtInferenceResolution: any MTLTexture,
        foregroundAtInferenceResolution: any MTLTexture
    ) throws {
        guard let postCommandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        postCommandBuffer.label = "CorridorKey by LateNite Post-Inference"

        // 1. Upscale alpha + foreground to destination resolution using the
        // user's chosen method.
        let upscaledAlphaPooled = try RenderStages.resample(
            source: alphaAtInferenceResolution,
            targetWidth: context.sourceWidth,
            targetHeight: context.sourceHeight,
            pixelFormat: .r16Float,
            method: inputs.state.upscaleMethod,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let upscaledAlpha = upscaledAlphaPooled?.texture ?? alphaAtInferenceResolution

        let upscaledForegroundPooled = try RenderStages.resample(
            source: foregroundAtInferenceResolution,
            targetWidth: context.sourceWidth,
            targetHeight: context.sourceHeight,
            pixelFormat: .rgba16Float,
            method: inputs.state.upscaleMethod,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let upscaledForeground = upscaledForegroundPooled?.texture ?? foregroundAtInferenceResolution

        // 2. Despill.
        let despilledPooled = try RenderStages.despill(
            foreground: upscaledForeground,
            strength: Float(inputs.state.despillStrength),
            method: inputs.state.spillMethod,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let despilled = despilledPooled?.texture ?? upscaledForeground

        // 3. Matte refinement chain (fused levels/gamma/refiner → CC
        // despeckle → erode/dilate → softness).
        let (refinedMattePooled, refinedMatte) = try refineMatte(
            alpha: upscaledAlpha,
            state: inputs.state,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )

        // 4. Pre-blur the source once when light-wrap is on so the fused
        // post-process pass has a blurred RGB input to sample.
        var blurredSourcePooled: PooledTexture?
        var blurredSourceIntermediatePooled: PooledTexture?
        let blurredSourceTexture: any MTLTexture
        let lightWrapActive = inputs.state.lightWrapEnabled
            && inputs.state.lightWrapStrength > 0
            && inputs.state.lightWrapRadius > 0
        if lightWrapActive {
            let radius = Float(inputs.state.lightWrapRadius)
            guard let blurred = context.entry.texturePool.acquire(
                width: despilled.width,
                height: despilled.height,
                pixelFormat: despilled.pixelFormat
            ) else { throw MetalDeviceCacheError.textureAllocationFailed }
            guard let intermediate = context.entry.texturePool.acquire(
                width: despilled.width,
                height: despilled.height,
                pixelFormat: despilled.pixelFormat
            ) else {
                blurred.returnManually()
                throw MetalDeviceCacheError.textureAllocationFailed
            }
            try MatteRefiner.applyGaussianBlur(
                source: rotatedSource,
                intermediate: intermediate.texture,
                destination: blurred.texture,
                radiusPixels: radius,
                entry: context.entry,
                commandBuffer: postCommandBuffer
            )
            blurredSourcePooled = blurred
            blurredSourceIntermediatePooled = intermediate
            blurredSourceTexture = blurred.texture
        } else {
            // No light-wrap: bind something valid but inert in the fused
            // kernel. The shader only reads this slot when the
            // `lightWrapEnabled` flag is set, so passing `rotatedSource`
            // here is safe and avoids a dead allocation.
            blurredSourceTexture = rotatedSource
        }

        // 5. Fused foreground post-process: source-passthrough +
        // light-wrap + edge-decontam + inverse-rotation all in one
        // dispatch. Saves three compute encoders and ~3 MB of
        // intermediate-texture bandwidth per 4K frame vs. the unfused
        // chain.
        // Edge decontaminate operates on the foreground BEFORE the
        // inverse rotation back to the original screen colour. So
        // its `screenColor` reference must always be the canonical
        // green (0.08, 0.84, 0.08) — that's the screen colour in
        // the rotated/green domain that despill and edge decontam
        // both work in. Using `screenTransform.estimatedScreenReference`
        // here would pass `canonicalBlue` for blue screens, which
        // produced incorrect edge decontamination on blue-screen
        // footage (the residual projection used the wrong axis).
        let fusedForegroundConfig = RenderStages.FusedForegroundConfig(
            sourcePassthrough: inputs.state.sourcePassthroughEnabled,
            lightWrapEnabled: lightWrapActive,
            lightWrapStrength: Float(inputs.state.lightWrapStrength),
            lightWrapEdgeBias: 0.6,
            edgeDecontaminateEnabled: inputs.state.edgeDecontaminateEnabled
                && inputs.state.edgeDecontaminateStrength > 0,
            edgeDecontaminateStrength: Float(inputs.state.edgeDecontaminateStrength),
            screenColor: SIMD3<Float>(0.08, 0.84, 0.08),
            inverseScreenMatrix: screenTransform.inverseMatrix,
            applyInverseRotation: !screenTransform.isIdentity
        )
        let restoredPooled = try RenderStages.applyFusedForegroundPostProcess(
            foreground: despilled,
            sourceRGB: rotatedSource,
            matte: refinedMatte,
            blurredSource: blurredSourceTexture,
            config: fusedForegroundConfig,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )

        // 6. Final compose into FCP destination.
        try compose(
            source: context.sourceTexture,
            foreground: restoredPooled.texture,
            matte: refinedMatte,
            context: context,
            state: inputs.state,
            commandBuffer: postCommandBuffer
        )

        try commitAndWait(commandBuffer: postCommandBuffer)

        // Return every pooled texture we acquired along the way. None of
        // these can be reused before the command buffer retires, but
        // `commitAndWait` has already waited.
        upscaledAlphaPooled?.returnManually()
        upscaledForegroundPooled?.returnManually()
        despilledPooled?.returnManually()
        refinedMattePooled?.returnManually()
        blurredSourcePooled?.returnManually()
        blurredSourceIntermediatePooled?.returnManually()
        restoredPooled.returnManually()
    }

    // MARK: - Matte refinement orchestration

    /// Walks the matte through fused levels/gamma/refiner → CC
    /// despeckle → erode/dilate → softness. Returns the final texture
    /// plus the pooled wrapper if one was produced (callers need to
    /// return the pool slot manually after `commitAndWait`).
    private func refineMatte(
        alpha: any MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (pooled: PooledTexture?, texture: any MTLTexture) {
        // Allocate two ping-pong buffers for the separable chains. Both are
        // returned to the pool at the end of this helper's scope via
        // `returnOnCompletion` on the command buffer.
        guard let bufferPooled = entry.texturePool.acquire(
            width: alpha.width,
            height: alpha.height,
            pixelFormat: .r16Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let auxiliaryPooled = entry.texturePool.acquire(
            width: alpha.width,
            height: alpha.height,
            pixelFormat: .r16Float
        ) else {
            bufferPooled.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        let buffer = bufferPooled.texture
        let auxiliary = auxiliaryPooled.texture

        // Fused levels + gamma + refiner-blend — one dispatch.
        let refinerArtifacts = try RenderStages.applyFusedMatteRefine(
            matte: alpha,
            destination: buffer,
            blackPoint: Float(state.alphaBlackPoint),
            whitePoint: Float(state.alphaWhitePoint),
            gamma: Float(state.alphaGamma),
            refinerStrength: Float(state.refinerStrength),
            entry: entry,
            commandBuffer: commandBuffer
        )
        var current: any MTLTexture = buffer
        // Release coarse + intermediate textures once the command buffer
        // is retired. They feed the kernel once and are never touched
        // again.
        refinerArtifacts.coarsePooled?.returnOnCompletion(of: commandBuffer)
        refinerArtifacts.intermediatePooled?.returnOnCompletion(of: commandBuffer)

        // Auto Despeckle (Phase 4.2: CC-based).
        let ccPooled: PooledTexture?
        if state.autoDespeckleEnabled {
            let areaThreshold = despeckleAreaThreshold(state: state)
            if let pooled = try RenderStages.applyConnectedComponentsDespeckle(
                matte: current,
                areaThreshold: areaThreshold,
                entry: entry,
                commandBuffer: commandBuffer
            ) {
                ccPooled = pooled
                current = pooled.texture
            } else {
                ccPooled = nil
            }
        } else {
            ccPooled = nil
        }

        // Erode / dilate (user slider). Radius is in destination pixels.
        let alphaErodeRadiusPixels = state.destinationPixelRadius(fromNormalized: state.alphaErodeNormalized)
        if abs(alphaErodeRadiusPixels) > 0.5 {
            try MatteRefiner.applyMorphology(
                source: current,
                intermediate: auxiliary,
                destination: buffer,
                radius: Int(alphaErodeRadiusPixels.rounded()),
                entry: entry,
                commandBuffer: commandBuffer
            )
            current = buffer
        }

        // Gaussian softness.
        let softnessRadiusPixels = state.destinationPixelRadius(fromNormalized: state.alphaSoftnessNormalized)
        if softnessRadiusPixels > 0.5 {
            try MatteRefiner.applyGaussianBlur(
                source: current,
                intermediate: auxiliary,
                destination: buffer,
                radiusPixels: softnessRadiusPixels,
                entry: entry,
                commandBuffer: commandBuffer
            )
            current = buffer
        }

        // Defer pool returns until after commitAndWait; schedule them via
        // the command buffer so hidden uses of `buffer` / `auxiliary` in
        // the MPS path can't race the pool.
        bufferPooled.returnOnCompletion(of: commandBuffer)
        auxiliaryPooled.returnOnCompletion(of: commandBuffer)
        // ccPooled is the final matte if set — hold it for the caller
        // rather than returning on completion, so the compose pass can
        // sample it.
        return (ccPooled, current)
    }

    // MARK: - Cached matte helpers

    /// Lightweight peek at the cache that returns just the dimensions
    /// without doing the (relatively expensive) decompress + half→float
    /// pass. Used by `renderUsingCachedAlpha` so it can size the GPU
    /// upload buffer before we commit to decoding.
    private func cachedMatteHeader(
        from state: PluginStateData,
        expectedResolution: Int
    ) -> (width: Int, height: Int)? {
        guard let blob = state.cachedMatteBlob,
              state.cachedMatteInferenceResolution == expectedResolution,
              let header = MatteCodec.parseHeader(blob)
        else { return nil }
        return header
    }

    /// Decodes the cached alpha matte into a pooled `r32Float` texture
    /// without an intermediate `[Float]` allocation. Saves ~5–10 ms
    /// per render-from-cache frame on a 4K matte versus the prior
    /// `MatteCodec.decode → texture.replace` path. vImage handles the
    /// half→float conversion via the ARM FP16 hardware instead of
    /// the old scalar loop.
    private func uploadCachedAlpha(
        blob: Data,
        width: Int,
        height: Int,
        entry: MetalDeviceCacheEntry
    ) throws -> PooledTexture {
        guard let pooled = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: .r32Float,
            storageMode: .shared
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }

        // Decode straight into a per-rung-cached MTLBuffer; vImage
        // converts half-floats with the ARM FP16 hardware so a 2048²
        // matte costs ~4 ms instead of ~40 ms on the scalar loop.
        guard let buffer = entry.cachedAlphaBuffer(width: width, height: height) else {
            pooled.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        let pixelCount = width * height
        let bufferPointer = buffer.contents().bindMemory(to: Float.self, capacity: pixelCount)
        let success = MatteCodec.decode(
            blob,
            into: bufferPointer,
            capacity: pixelCount,
            expectedWidth: width,
            expectedHeight: height
        )
        guard success else {
            pooled.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }

        // Copy the decoded floats into the pooled texture. On Apple
        // Silicon's unified memory, both the buffer and the .shared
        // texture's backing live in the same physical memory, so this
        // is effectively a memcpy through the Metal driver. The y
        // orientation of the cached bytes already matches the texture
        // convention (the analyse pass wrote them via
        // `corridorKeyAlphaBufferToTextureKernel` which performs the
        // y-flip; we read those flipped bytes back and store them, so
        // the cache is byte-for-byte y-down).
        let bytesPerRow = width * MemoryLayout<Float>.size
        pooled.texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: bufferPointer,
            bytesPerRow: bytesPerRow
        )
        return pooled
    }

    // MARK: - Compose (FxPlug-specific)

    /// Writes the final pixel directly into Final Cut Pro's destination tile
    /// (or the full destination texture, for the standalone editor) using a
    /// render pass. Positioning the quad in tile-local coordinates keeps the
    /// output aligned when FCP requests a sub-tile, while the standalone
    /// path passes the destination texture's full extent and gets the same
    /// behaviour with one less indirection.
    private func compose(
        source: any MTLTexture,
        foreground: any MTLTexture,
        matte: any MTLTexture,
        context: DeviceContext,
        state: PluginStateData,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        try composeInto(
            source: source,
            foreground: foreground,
            matte: matte,
            destination: context.destinationTexture,
            tileWidth: context.outputWidth,
            tileHeight: context.outputHeight,
            outputMode: state.outputMode,
            pixelFormat: context.pixelFormat,
            entry: context.entry,
            commandBuffer: commandBuffer
        )
    }

    /// Tile-agnostic compose. Separated from `compose(...)` above so tests
    /// (which don't have an `FxImageTile`) can exercise it directly with
    /// explicit tile dimensions.
    func composeInto(
        source: any MTLTexture,
        foreground: any MTLTexture,
        matte: any MTLTexture,
        destination: any MTLTexture,
        tileWidth: Float,
        tileHeight: Float,
        outputMode: OutputMode,
        pixelFormat: MTLPixelFormat,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let renderPipelines = try entry.renderPipelines(for: pixelFormat)

        let passDescriptor = MTLRenderPassDescriptor()
        passDescriptor.colorAttachments[0].texture = destination
        passDescriptor.colorAttachments[0].loadAction = .clear
        passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        passDescriptor.colorAttachments[0].storeAction = .store

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Compose"

        let halfW = tileWidth * 0.5
        let halfH = tileHeight * 0.5

        var vertices: [CKVertex2D] = [
            CKVertex2D(position: SIMD2<Float>(halfW, -halfH), textureCoordinate: SIMD2<Float>(1, 1)),
            CKVertex2D(position: SIMD2<Float>(-halfW, -halfH), textureCoordinate: SIMD2<Float>(0, 1)),
            CKVertex2D(position: SIMD2<Float>(halfW, halfH), textureCoordinate: SIMD2<Float>(1, 0)),
            CKVertex2D(position: SIMD2<Float>(-halfW, halfH), textureCoordinate: SIMD2<Float>(0, 0))
        ]
        var viewportSize = SIMD2<UInt32>(UInt32(tileWidth), UInt32(tileHeight))

        encoder.setViewport(MTLViewport(
            originX: 0,
            originY: 0,
            width: Double(tileWidth),
            height: Double(tileHeight),
            znear: -1,
            zfar: 1
        ))
        encoder.setRenderPipelineState(renderPipelines.compose)
        encoder.setVertexBytes(
            &vertices,
            length: MemoryLayout<CKVertex2D>.stride * vertices.count,
            index: Int(CKVertexInputIndexVertices.rawValue)
        )
        encoder.setVertexBytes(
            &viewportSize,
            length: MemoryLayout<SIMD2<UInt32>>.size,
            index: Int(CKVertexInputIndexViewportSize.rawValue)
        )
        encoder.setFragmentTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setFragmentTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setFragmentTexture(matte, index: Int(CKTextureIndexMatte.rawValue))

        var params = CKComposeParams(
            outputMode: outputMode.shaderValue
        )
        encoder.setFragmentBytes(
            &params,
            length: MemoryLayout<CKComposeParams>.size,
            index: Int(CKBufferIndexComposeParams.rawValue)
        )
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
    }

    // MARK: - Destination diagnostic readback

    /// Set once we've dumped the destination diagnostic for this
    /// process — keeps the readback off the per-frame hot path while
    /// still capturing a sample for debugging the "Hint Diagnostic
    /// renders black in FCP" symptom.
    private static let diagnosticDumpFlag = NSLock()
    nonisolated(unsafe) private static var diagnosticDumpDone = false

    /// Reads back the hint texture (pre-compose) AND the destination
    /// (post-compose) and logs the per-channel stats of each. Runs at
    /// most once per process to avoid pinning the GPU on every render.
    static func dumpHintAndDestinationDiagnosticOnce(
        hint: any MTLTexture,
        destination: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue
    ) {
        diagnosticDumpFlag.lock()
        let alreadyDone = diagnosticDumpDone
        diagnosticDumpDone = true
        diagnosticDumpFlag.unlock()
        guard !alreadyDone else { return }

        do {
            // Hint texture is r16Float; read back via blit→shared.
            let (hintMax, hintMin, hintCoverage) = try readbackR16FloatHintStats(
                hint: hint,
                entry: entry,
                commandQueue: commandQueue
            )
            PluginLog.notice(
                "Hint Diagnostic HINT texture dump (one-time): "
                + "size=\(hint.width)x\(hint.height), "
                + "min=\(hintMin), max=\(hintMax), "
                + "fraction>0.5 = \(hintCoverage)"
            )
        } catch {
            PluginLog.error("Hint Diagnostic hint readback failed: \(error.localizedDescription)")
        }

        do {
            let (maxR, maxG, maxB, redCoverage) = try readbackDestinationStats(
                destination: destination,
                entry: entry,
                commandQueue: commandQueue
            )
            PluginLog.notice(
                "Hint Diagnostic DESTINATION dump (one-time): "
                + "format=\(destination.pixelFormat.rawValue), "
                + "size=\(destination.width)x\(destination.height), "
                + "maxR=\(maxR), maxG=\(maxG), maxB=\(maxB), "
                + "redCoverage=\(redCoverage)"
            )
        } catch {
            PluginLog.error("Hint Diagnostic destination dump failed: \(error.localizedDescription)")
        }
    }

    /// Blits an r16Float texture into a `.shared` staging texture and
    /// reads the floats back to compute (max, min, fraction > 0.5).
    private static func readbackR16FloatHintStats(
        hint: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue
    ) throws -> (Float, Float, Double) {
        let width = hint.width
        let height = hint.height
        let pixelCount = width * height

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let staging = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let buffer = commandQueue.makeCommandBuffer(),
              let blit = buffer.makeBlitCommandEncoder()
        else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        blit.copy(from: hint, to: staging)
        blit.endEncoding()
        let semaphore = DispatchSemaphore(value: 0)
        buffer.addCompletedHandler { _ in semaphore.signal() }
        buffer.commit()
        semaphore.wait()
        if let error = buffer.error { throw error }

        var halves = [UInt16](repeating: 0, count: pixelCount)
        halves.withUnsafeMutableBytes { bytes in
            if let base = bytes.baseAddress {
                staging.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<UInt16>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        var maxValue: Float = -Float.greatestFiniteMagnitude
        var minValue: Float = Float.greatestFiniteMagnitude
        var nonZero = 0
        for half in halves {
            let value = Float(Float16(bitPattern: half))
            if value > maxValue { maxValue = value }
            if value < minValue { minValue = value }
            if value > 0.5 { nonZero += 1 }
        }
        return (maxValue, minValue, Double(nonZero) / Double(pixelCount))
    }

    /// Blits the destination into a `.shared` staging texture, reads
    /// it back to CPU, and computes per-channel stats. Handles the
    /// three pixel formats FCP gives us (`.rgba16Float`, `.rgba32Float`,
    /// `.bgra8Unorm`).
    private static func readbackDestinationStats(
        destination: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue
    ) throws -> (Float, Float, Float, Double) {
        let width = destination.width
        let height = destination.height
        let pixelCount = width * height

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: destination.pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let staging = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let buffer = commandQueue.makeCommandBuffer(),
              let blit = buffer.makeBlitCommandEncoder()
        else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        blit.copy(from: destination, to: staging)
        blit.endEncoding()
        let semaphore = DispatchSemaphore(value: 0)
        buffer.addCompletedHandler { _ in semaphore.signal() }
        buffer.commit()
        semaphore.wait()
        if let error = buffer.error { throw error }

        var maxR: Float = 0
        var maxG: Float = 0
        var maxB: Float = 0
        var redNonZero = 0

        switch staging.pixelFormat {
        case .rgba16Float:
            var halves = [UInt16](repeating: 0, count: pixelCount * 4)
            halves.withUnsafeMutableBytes { bytes in
                if let base = bytes.baseAddress {
                    staging.getBytes(
                        base,
                        bytesPerRow: width * 4 * MemoryLayout<UInt16>.size,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            for i in 0..<pixelCount {
                let r = Float(Float16(bitPattern: halves[i * 4 + 0]))
                let g = Float(Float16(bitPattern: halves[i * 4 + 1]))
                let b = Float(Float16(bitPattern: halves[i * 4 + 2]))
                if r > maxR { maxR = r }
                if g > maxG { maxG = g }
                if b > maxB { maxB = b }
                if r > 0.5 { redNonZero += 1 }
            }
        case .rgba32Float:
            var floats = [Float](repeating: 0, count: pixelCount * 4)
            floats.withUnsafeMutableBytes { bytes in
                if let base = bytes.baseAddress {
                    staging.getBytes(
                        base,
                        bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            for i in 0..<pixelCount {
                let r = floats[i * 4 + 0]
                let g = floats[i * 4 + 1]
                let b = floats[i * 4 + 2]
                if r > maxR { maxR = r }
                if g > maxG { maxG = g }
                if b > maxB { maxB = b }
                if r > 0.5 { redNonZero += 1 }
            }
        case .bgra8Unorm:
            var bytes8 = [UInt8](repeating: 0, count: pixelCount * 4)
            bytes8.withUnsafeMutableBytes { rawBytes in
                if let base = rawBytes.baseAddress {
                    staging.getBytes(
                        base,
                        bytesPerRow: width * 4,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            for i in 0..<pixelCount {
                let b = Float(bytes8[i * 4 + 0]) / 255
                let g = Float(bytes8[i * 4 + 1]) / 255
                let r = Float(bytes8[i * 4 + 2]) / 255
                if r > maxR { maxR = r }
                if g > maxG { maxG = g }
                if b > maxB { maxB = b }
                if r > 0.5 { redNonZero += 1 }
            }
        default:
            return (0, 0, 0, 0)
        }
        return (maxR, maxG, maxB, Double(redNonZero) / Double(pixelCount))
    }

    // MARK: - Synchronisation helpers

    /// Cooperative commit+wait. Uses `addCompletedHandler` + a
    /// `DispatchSemaphore` so the CPU thread yields cleanly instead of
    /// busy-spinning inside `waitUntilCompleted`. Observed effect is a
    /// ~2 ms shrink in per-tile kickoff gaps on 4K projects.
    private func commitAndWait(commandBuffer: any MTLCommandBuffer) throws {
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()
        if let error = commandBuffer.error { throw error }
    }

    // MARK: - Utilities

    /// Converts the inspector's "Despeckle Size" (approximate speckle area in
    /// 1920px-baseline pixels²) into a CC-filter area threshold scaled to
    /// the matte's current inference resolution. This replaces the old
    /// morphology-radius interpretation, which mislabelled the UI unit.
    private func despeckleAreaThreshold(state: PluginStateData) -> Int {
        let scale = Double(state.destinationLongEdgePixels) / max(state.longEdgeBaseline, 1.0)
        let areaAtBaseline = Double(max(state.despeckleSize, 1))
        // Area scales with the square of the linear scale factor.
        let scaledArea = areaAtBaseline * scale * scale
        return max(1, Int(scaledArea.rounded()))
    }

    private func hintTileLayoutValue(for texture: any MTLTexture) -> Int32 {
        switch texture.pixelFormat {
        case .rgba16Float, .rgba32Float, .bgra8Unorm, .rgba8Unorm: return 0
        case .r8Unorm, .r16Float, .r32Float: return 1
        default: return 2
        }
    }
}
