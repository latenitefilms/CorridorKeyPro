//
//  RenderPipeline.swift
//  Corridor Key Toolbox
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

/// Input bundle for a render. Value-type keeps the orchestrator easy to reason
/// about; `FxImageTile` is passed along because we only read it, never mutate.
struct RenderRequest: @unchecked Sendable {
    let destinationImage: FxImageTile
    let sourceImage: FxImageTile
    let alphaHintImage: FxImageTile?
    let state: PluginStateData
    let workingGamut: WorkingColorGamut
    let renderTime: CMTime
}

/// Result fed back to the FxPlug layer so it can surface status in the
/// inspector's "Runtime Status" group.
struct RenderReport: Sendable {
    let backendDescription: String
    let guideSourceDescription: String
    let effectiveInferenceResolution: Int
    let deviceName: String
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

    /// Executes the full render for one tile. Returns a `RenderReport` so the
    /// FxPlug layer can update the runtime status fields.
    func render(_ request: RenderRequest) throws -> RenderReport {
        let context = try makeDeviceContext(for: request)
        defer { context.entry.returnCommandQueue(context.commandQueue) }

        let screenTransform = ScreenColorEstimator.defaultTransform(for: request.state.screenColor)
        let gamutTransform = ColorGamutMatrix.transform(for: request.workingGamut)
        let inferenceResolution = request.state.qualityMode.resolvedInferenceResolution(
            forLongEdge: context.longEdge
        )

        // Fast path: FxAnalyzer already ran for this clip and the inference
        // resolution matches the quality the user is asking for. Skip MLX
        // entirely — the cache already holds a network-quality matte.
        if let cached = decodedCachedMatte(from: request.state, expectedResolution: inferenceResolution) {
            return try renderUsingCachedAlpha(
                request: request,
                context: context,
                screenTransform: screenTransform,
                gamutTransform: gamutTransform,
                inferenceResolution: inferenceResolution,
                cachedAlpha: cached
            )
        }
        // Unanalysed → leave the source untouched. Running MLX on the render
        // thread made toggling the effect feel laggy and produced inconsistent
        // output while the analysis cache was being built, so pass-through is
        // now the explicit "nothing to key yet" signal.
        return try renderSourcePassThrough(request: request, context: context)
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
        let screenTransform = ScreenColorEstimator.defaultTransform(for: state.screenColor)
        let gamutTransform = ColorGamutMatrix.transform(for: workingGamut)
        let longEdge = max(sourceTexture.width, sourceTexture.height)
        let inferenceResolution = state.qualityMode.resolvedInferenceResolution(forLongEdge: longEdge)

        let pre = try runPreInference(
            sourceTexture: sourceTexture,
            hintTile: nil,
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
    }

    private struct PreInferenceArtifacts {
        let rotatedSource: PooledTexture?
        let normalisedInputBuffer: any MTLBuffer
        let rawSourceAtInferenceResolution: PooledTexture
    }

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
        return DeviceContext(
            device: device,
            entry: entry,
            commandQueue: commandQueue,
            sourceTexture: sourceTexture,
            destinationTexture: destinationTexture,
            pixelFormat: pixelFormat,
            sourceWidth: sourceTexture.width,
            sourceHeight: sourceTexture.height,
            longEdge: max(sourceTexture.width, sourceTexture.height)
        )
    }

    /// Writes the source tile straight through to the destination. Used when
    /// the clip hasn't been analysed yet — the plug-in stays out of the way
    /// until the cache is populated.
    private func renderSourcePassThrough(
        request: RenderRequest,
        context: DeviceContext
    ) throws -> RenderReport {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        commandBuffer.label = "Corridor Key Toolbox Pass-Through"

        // Reuse the compose render pipeline with the source bound as every
        // sampler and `foregroundOnly` output. This re-publishes the raw
        // source bytes through a tiny render pass so format and tile-layout
        // conversions the host expects still happen, without touching MLX.
        var passThroughState = request.state
        passThroughState.outputMode = .foregroundOnly
        try compose(
            source: context.sourceTexture,
            foreground: context.sourceTexture,
            matte: context.sourceTexture,
            destination: context.destinationTexture,
            destinationTile: request.destinationImage,
            state: passThroughState,
            pixelFormat: context.pixelFormat,
            entry: context.entry,
            commandBuffer: commandBuffer
        )

        try commitAndWait(commandBuffer: commandBuffer)

        return RenderReport(
            backendDescription: "Source Pass-Through",
            guideSourceDescription: "Clip not analysed",
            effectiveInferenceResolution: 0,
            deviceName: context.device.name
        )
    }

    private func renderUsingCachedAlpha(
        request: RenderRequest,
        context: DeviceContext,
        screenTransform: ScreenColorTransform,
        gamutTransform: WorkingSpaceTransform,
        inferenceResolution: Int,
        cachedAlpha: (alpha: [Float], width: Int, height: Int)
    ) throws -> RenderReport {
        guard let preCommandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        preCommandBuffer.label = "Corridor Key Toolbox Cached Pre-Inference"

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
            method: request.state.upscaleMethod,
            entry: context.entry,
            commandBuffer: preCommandBuffer
        )
        let rawSourceAtInferenceResolution = rawSourcePooled?.texture ?? rotatedSource

        try commitAndWait(commandBuffer: preCommandBuffer)

        rotatedSourcePooled?.returnManually()
        // rawSourcePooled stays alive — we need it as the foreground stand-in.

        let alphaPooled = try uploadCachedAlpha(
            alpha: cachedAlpha.alpha,
            width: cachedAlpha.width,
            height: cachedAlpha.height,
            entry: context.entry
        )

        try runPostInference(
            request: request,
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
        hintTile: FxImageTile?,
        device: any MTLDevice,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue,
        screenTransform: ScreenColorTransform,
        gamutTransform: WorkingSpaceTransform,
        inferenceResolution: Int
    ) throws -> PreInferenceArtifacts {
        guard let preCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        preCommandBuffer.label = "Corridor Key Toolbox Pre-Inference"

        let rotatedSourcePooled = try RenderStages.applyScreenMatrix(
            source: sourceTexture,
            matrix: screenTransform.forwardMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: entry,
            commandBuffer: preCommandBuffer
        )
        let rotatedSource = rotatedSourcePooled?.texture ?? sourceTexture

        let hintTexturePooled: PooledTexture
        if let hintTile, let hostTexture = hintTile.metalTexture(for: device) {
            hintTexturePooled = try RenderStages.extractHint(
                source: hostTexture,
                layout: hintTileLayoutValue(for: hostTexture),
                targetWidth: rotatedSource.width,
                targetHeight: rotatedSource.height,
                entry: entry,
                commandBuffer: preCommandBuffer
            )
        } else {
            hintTexturePooled = try RenderStages.generateGreenHint(
                source: rotatedSource,
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
                blit.label = "Corridor Key Toolbox Pre-Inf Copy"
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
        request: RenderRequest,
        context: DeviceContext,
        screenTransform: ScreenColorTransform,
        rotatedSource: any MTLTexture,
        alphaAtInferenceResolution: any MTLTexture,
        foregroundAtInferenceResolution: any MTLTexture
    ) throws {
        guard let postCommandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        postCommandBuffer.label = "Corridor Key Toolbox Post-Inference"

        // 1. Upscale alpha + foreground to destination resolution using the
        // user's chosen method.
        let upscaledAlphaPooled = try RenderStages.resample(
            source: alphaAtInferenceResolution,
            targetWidth: context.sourceWidth,
            targetHeight: context.sourceHeight,
            pixelFormat: .r16Float,
            method: request.state.upscaleMethod,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let upscaledAlpha = upscaledAlphaPooled?.texture ?? alphaAtInferenceResolution

        let upscaledForegroundPooled = try RenderStages.resample(
            source: foregroundAtInferenceResolution,
            targetWidth: context.sourceWidth,
            targetHeight: context.sourceHeight,
            pixelFormat: .rgba16Float,
            method: request.state.upscaleMethod,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let upscaledForeground = upscaledForegroundPooled?.texture ?? foregroundAtInferenceResolution

        // 2. Despill.
        let despilledPooled = try RenderStages.despill(
            foreground: upscaledForeground,
            strength: Float(request.state.despillStrength),
            method: request.state.spillMethod,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let despilled = despilledPooled?.texture ?? upscaledForeground

        // 3. Matte refinement chain (fused levels/gamma/refiner → CC
        // despeckle → erode/dilate → softness).
        let (refinedMattePooled, refinedMatte) = try refineMatte(
            alpha: upscaledAlpha,
            state: request.state,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )

        // 4. Pre-blur the source once when light-wrap is on so the fused
        // post-process pass has a blurred RGB input to sample.
        var blurredSourcePooled: PooledTexture?
        var blurredSourceIntermediatePooled: PooledTexture?
        let blurredSourceTexture: any MTLTexture
        let lightWrapActive = request.state.lightWrapEnabled
            && request.state.lightWrapStrength > 0
            && request.state.lightWrapRadius > 0
        if lightWrapActive {
            let radius = Float(request.state.lightWrapRadius)
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
        let fusedForegroundConfig = RenderStages.FusedForegroundConfig(
            sourcePassthrough: request.state.sourcePassthroughEnabled,
            lightWrapEnabled: lightWrapActive,
            lightWrapStrength: Float(request.state.lightWrapStrength),
            lightWrapEdgeBias: 0.6,
            edgeDecontaminateEnabled: request.state.edgeDecontaminateEnabled
                && request.state.edgeDecontaminateStrength > 0,
            edgeDecontaminateStrength: Float(request.state.edgeDecontaminateStrength),
            screenColor: screenTransform.estimatedScreenReference,
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
            destination: context.destinationTexture,
            destinationTile: request.destinationImage,
            state: request.state,
            pixelFormat: context.pixelFormat,
            entry: context.entry,
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

    private func decodedCachedMatte(
        from state: PluginStateData,
        expectedResolution: Int
    ) -> (alpha: [Float], width: Int, height: Int)? {
        guard let blob = state.cachedMatteBlob,
              state.cachedMatteInferenceResolution == expectedResolution,
              let decoded = MatteCodec.decode(blob),
              decoded.width > 0, decoded.height > 0
        else {
            return nil
        }
        return decoded
    }

    private func uploadCachedAlpha(
        alpha: [Float],
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
        // The cached alpha was written by `corridorKeyAlphaBufferToTextureKernel`
        // (during the analysis pass), which already flipped the MLX bridge's
        // y-up output into the y-down convention the rest of the pipeline
        // uses. We then read those bytes back via `getBytes` and stored them
        // in the FCP custom-parameter cache — so the cached buffer is
        // **already y-down** and we upload it byte-for-byte.
        //
        // Earlier builds also flipped the rows here, which was correct
        // pre-zero-copy (when the writeback was a CPU memcpy that didn't
        // flip). Post-zero-copy that became a double-flip and produced an
        // upside-down matte at compose time. See `MatteOrientationTests` in
        // `CorridorKeyToolboxInferenceTests` for a regression gate.
        let bytesPerRow = width * MemoryLayout<Float>.size
        alpha.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                pooled.texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: bytesPerRow
                )
            }
        }
        return pooled
    }

    // MARK: - Compose (FxPlug-specific)

    /// Writes the final pixel directly into Final Cut Pro's destination tile
    /// using a render pass. Positioning the quad in tile-local coordinates
    /// keeps the output aligned when FCP requests a sub-tile.
    private func compose(
        source: any MTLTexture,
        foreground: any MTLTexture,
        matte: any MTLTexture,
        destination: any MTLTexture,
        destinationTile: FxImageTile,
        state: PluginStateData,
        pixelFormat: MTLPixelFormat,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let outputWidth = Float(destinationTile.tilePixelBounds.right - destinationTile.tilePixelBounds.left)
        let outputHeight = Float(destinationTile.tilePixelBounds.top - destinationTile.tilePixelBounds.bottom)
        try composeInto(
            source: source,
            foreground: foreground,
            matte: matte,
            destination: destination,
            tileWidth: outputWidth,
            tileHeight: outputHeight,
            outputMode: state.outputMode,
            pixelFormat: pixelFormat,
            entry: entry,
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
        encoder.label = "Corridor Key Toolbox Compose"

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
