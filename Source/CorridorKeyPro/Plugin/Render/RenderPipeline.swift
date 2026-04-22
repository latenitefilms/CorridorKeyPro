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
//  3. Post-inference — upscale → despill → matte refine → passthrough →
//     restore → compose directly into Final Cut Pro's destination texture
//     via a render pass.
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

    /// Runs pre-inference + MLX (no post-processing) and returns the raw alpha
    /// matte on the CPU. Called by `FxAnalyzer.analyzeFrame` so the matte can
    /// be compressed and stored in the custom parameter. This path never reads
    /// from or writes to the per-frame cache — the custom parameter is the
    /// persistent cache.
    func extractAlphaMatteForAnalysis(
        sourceTexture: any MTLTexture,
        state: PluginStateData,
        renderTime: CMTime,
        device: any MTLDevice,
        entry: MetalDeviceCacheEntry,
        commandQueue: any MTLCommandQueue
    ) throws -> (alpha: [Float], width: Int, height: Int, inferenceResolution: Int) {
        let screenTransform = ScreenColorEstimator.defaultTransform(for: state.screenColor)
        let longEdge = max(sourceTexture.width, sourceTexture.height)
        let inferenceResolution = state.qualityMode.resolvedInferenceResolution(forLongEdge: longEdge)

        let pre = try runPreInference(
            sourceTexture: sourceTexture,
            hintTile: nil,
            device: device,
            entry: entry,
            commandQueue: commandQueue,
            screenTransform: screenTransform,
            inferenceResolution: inferenceResolution
        )

        let cacheKey = InferenceCacheKey(
            frameTime: renderTime,
            screenColorRaw: state.screenColor.rawValue,
            inferenceResolution: inferenceResolution,
            cacheEntry: entry
        )
        let inferenceOutput = try inferenceCoordinator.runInference(
            request: KeyingInferenceRequest(
                normalisedInputTexture: pre.normalisedInput,
                rawSourceTexture: pre.rawSourceAtInferenceResolution,
                inferenceResolution: inferenceResolution
            ),
            cacheEntry: entry,
            cacheKey: cacheKey
        )

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
        return (alpha, width, height, inferenceResolution)
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
        let rotatedSource: any MTLTexture
        let normalisedInput: any MTLTexture
        let rawSourceAtInferenceResolution: any MTLTexture
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

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

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
        inferenceResolution: Int,
        cachedAlpha: (alpha: [Float], width: Int, height: Int)
    ) throws -> RenderReport {
        guard let preCommandBuffer = context.commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        preCommandBuffer.label = "Corridor Key Toolbox Cached Pre-Inference"

        let rotatedSource = try applyScreenRotation(
            source: context.sourceTexture,
            matrix: screenTransform.forwardMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: context.entry,
            commandBuffer: preCommandBuffer
        )
        let rawSourceAtInferenceResolution = try resample(
            source: rotatedSource,
            targetWidth: inferenceResolution,
            targetHeight: inferenceResolution,
            pixelFormat: .rgba16Float,
            entry: context.entry,
            commandBuffer: preCommandBuffer
        )
        preCommandBuffer.commit()
        preCommandBuffer.waitUntilCompleted()
        if let error = preCommandBuffer.error { throw error }

        let alphaTexture = try uploadCachedAlpha(
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
            alphaAtInferenceResolution: alphaTexture,
            foregroundAtInferenceResolution: rawSourceAtInferenceResolution
        )

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
        inferenceResolution: Int
    ) throws -> PreInferenceArtifacts {
        guard let preCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        preCommandBuffer.label = "Corridor Key Toolbox Pre-Inference"

        let rotatedSource = try applyScreenRotation(
            source: sourceTexture,
            matrix: screenTransform.forwardMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: entry,
            commandBuffer: preCommandBuffer
        )
        let hintTexture = try makeHintTexture(
            source: rotatedSource,
            hintTile: hintTile,
            device: device,
            entry: entry,
            commandBuffer: preCommandBuffer
        )
        let normalisedInput = try combineAndNormalise(
            source: rotatedSource,
            hint: hintTexture,
            inferenceResolution: inferenceResolution,
            entry: entry,
            commandBuffer: preCommandBuffer
        )
        let rawSourceAtInferenceResolution = try resample(
            source: rotatedSource,
            targetWidth: inferenceResolution,
            targetHeight: inferenceResolution,
            pixelFormat: .rgba16Float,
            entry: entry,
            commandBuffer: preCommandBuffer
        )
        preCommandBuffer.commit()
        preCommandBuffer.waitUntilCompleted()
        if let error = preCommandBuffer.error { throw error }

        return PreInferenceArtifacts(
            rotatedSource: rotatedSource,
            normalisedInput: normalisedInput,
            rawSourceAtInferenceResolution: rawSourceAtInferenceResolution
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

        let upscaledAlpha = try resample(
            source: alphaAtInferenceResolution,
            targetWidth: context.sourceWidth,
            targetHeight: context.sourceHeight,
            pixelFormat: .r16Float,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let upscaledForeground = try resample(
            source: foregroundAtInferenceResolution,
            targetWidth: context.sourceWidth,
            targetHeight: context.sourceHeight,
            pixelFormat: .rgba16Float,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let despilled = try despill(
            foreground: upscaledForeground,
            state: request.state,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let refinedMatte = try refineMatte(
            alpha: upscaledAlpha,
            state: request.state,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        let workingForeground: any MTLTexture
        if request.state.sourcePassthroughEnabled {
            workingForeground = try sourcePassthrough(
                foreground: despilled,
                source: rotatedSource,
                matte: refinedMatte,
                entry: context.entry,
                commandBuffer: postCommandBuffer
            )
        } else {
            workingForeground = despilled
        }
        let restoredForeground = try applyScreenRotation(
            source: workingForeground,
            matrix: screenTransform.inverseMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        try compose(
            source: context.sourceTexture,
            foreground: restoredForeground,
            matte: refinedMatte,
            destination: context.destinationTexture,
            destinationTile: request.destinationImage,
            state: request.state,
            pixelFormat: context.pixelFormat,
            entry: context.entry,
            commandBuffer: postCommandBuffer
        )
        postCommandBuffer.commit()
        postCommandBuffer.waitUntilCompleted()
        if let error = postCommandBuffer.error { throw error }
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
    ) throws -> any MTLTexture {
        guard let texture = entry.makeIntermediateTexture(
            width: width,
            height: height,
            pixelFormat: .r32Float,
            storageMode: .shared
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        // The bundled `.mlxfn` bridges output their tensor in `y-up` layout
        // — row 0 is the visual bottom, matching the CorridorKey-Runtime
        // OFX convention the bridges were exported against. The rest of
        // our pipeline (source IOSurface, compose shader) is `y-down`, so
        // we flip rows here so `matteTexture.sample(uv)` lines up pixel-
        // for-pixel with `sourceTexture.sample(uv)` in compose. Rough-matte
        // (pass-through) doesn't touch this path, so the non-MLX render
        // stays untouched.
        var flipped = [Float](repeating: 0, count: width * height)
        flipped.withUnsafeMutableBufferPointer { destPointer in
            alpha.withUnsafeBufferPointer { srcPointer in
                guard let destBase = destPointer.baseAddress,
                      let srcBase = srcPointer.baseAddress
                else { return }
                for row in 0..<height {
                    let srcRow = height - 1 - row
                    destBase
                        .advanced(by: row * width)
                        .update(from: srcBase.advanced(by: srcRow * width), count: width)
                }
            }
        }
        let bytesPerRow = width * MemoryLayout<Float>.size
        flipped.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: bytesPerRow
                )
            }
        }
        return texture
    }

    // MARK: - Stage helpers

    private func applyScreenRotation(
        source: any MTLTexture,
        matrix: simd_float3x3,
        isIdentity: Bool,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        if isIdentity { return source }
        guard let output = entry.makeIntermediateTexture(
            width: source.width,
            height: source.height,
            pixelFormat: .rgba16Float
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Screen Matrix"
        encoder.setComputePipelineState(entry.computePipelines.applyScreenMatrix)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output, index: Int(CKTextureIndexOutput.rawValue))
        var matrixCopy = matrix
        encoder.setBytes(
            &matrixCopy,
            length: MemoryLayout<simd_float3x3>.size,
            index: Int(CKBufferIndexScreenColorMatrix.rawValue)
        )
        dispatch(encoder: encoder, pipeline: entry.computePipelines.applyScreenMatrix, width: output.width, height: output.height)
        encoder.endEncoding()
        return output
    }

    private func makeHintTexture(
        source: any MTLTexture,
        hintTile: FxImageTile?,
        device: any MTLDevice,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        guard let hintTexture = entry.makeIntermediateTexture(
            width: source.width,
            height: source.height,
            pixelFormat: .r16Float
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Hint"

        if let hintTile, let hostTexture = hintTile.metalTexture(for: device) {
            encoder.setComputePipelineState(entry.computePipelines.extractHint)
            encoder.setTexture(hostTexture, index: Int(CKTextureIndexSource.rawValue))
            encoder.setTexture(hintTexture, index: Int(CKTextureIndexOutput.rawValue))
            var layout = hintTileLayoutValue(for: hostTexture)
            encoder.setBytes(&layout, length: MemoryLayout<Int32>.size, index: 0)
            dispatch(encoder: encoder, pipeline: entry.computePipelines.extractHint, width: hintTexture.width, height: hintTexture.height)
        } else {
            encoder.setComputePipelineState(entry.computePipelines.greenHint)
            encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
            encoder.setTexture(hintTexture, index: Int(CKTextureIndexOutput.rawValue))
            dispatch(encoder: encoder, pipeline: entry.computePipelines.greenHint, width: hintTexture.width, height: hintTexture.height)
        }
        encoder.endEncoding()
        return hintTexture
    }

    /// Downsamples source + hint to the inference resolution and writes the
    /// four-channel normalised tensor into a `.shared` storage texture so the
    /// inference engine can read it back from the CPU.
    private func combineAndNormalise(
        source: any MTLTexture,
        hint: any MTLTexture,
        inferenceResolution: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        guard let normalised = entry.makeIntermediateTexture(
            width: inferenceResolution,
            height: inferenceResolution,
            pixelFormat: .rgba32Float,
            storageMode: .shared
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Combine + Normalise"
        encoder.setComputePipelineState(entry.computePipelines.combineAndNormalize)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(hint, index: Int(CKTextureIndexHint.rawValue))
        encoder.setTexture(normalised, index: Int(CKTextureIndexOutput.rawValue))

        var params = CKNormalizeParams(
            mean: SIMD3<Float>(0.485, 0.456, 0.406),
            invStdDev: SIMD3<Float>(1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKNormalizeParams>.size,
            index: Int(CKBufferIndexNormalizeParams.rawValue)
        )
        dispatch(encoder: encoder, pipeline: entry.computePipelines.combineAndNormalize, width: inferenceResolution, height: inferenceResolution)
        encoder.endEncoding()
        return normalised
    }

    private func despill(
        foreground: any MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        guard state.despillStrength > 0 else { return foreground }
        guard let output = entry.makeIntermediateTexture(
            width: foreground.width,
            height: foreground.height
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Despill"
        encoder.setComputePipelineState(entry.computePipelines.despill)
        encoder.setTexture(foreground, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKDespillParams(
            strength: Float(state.despillStrength),
            method: state.spillMethod.shaderValue
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKDespillParams>.size,
            index: Int(CKBufferIndexDespillParams.rawValue)
        )
        dispatch(encoder: encoder, pipeline: entry.computePipelines.despill, width: output.width, height: output.height)
        encoder.endEncoding()
        return output
    }

    private func refineMatte(
        alpha: any MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        guard let buffer = entry.makeIntermediateTexture(
            width: alpha.width,
            height: alpha.height,
            pixelFormat: .r16Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let auxiliary = entry.makeIntermediateTexture(
            width: alpha.width,
            height: alpha.height,
            pixelFormat: .r16Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }

        var current = alpha

        try runAlphaLevelsGamma(source: current, destination: buffer, state: state, entry: entry, commandBuffer: commandBuffer)
        current = buffer

        if state.autoDespeckleEnabled {
            let despeckleRadius = despeckleRadiusPixels(state: state)
            if despeckleRadius > 0 {
                // Morphological open (erode → dilate) removes isolated specks up to
                // roughly `despeckleSize` pixels² without touching larger regions.
                try runMorphology(
                    source: current,
                    intermediate: auxiliary,
                    destination: buffer,
                    radius: -despeckleRadius,
                    entry: entry,
                    commandBuffer: commandBuffer
                )
                current = buffer
                try runMorphology(
                    source: current,
                    intermediate: auxiliary,
                    destination: buffer,
                    radius: despeckleRadius,
                    entry: entry,
                    commandBuffer: commandBuffer
                )
                current = buffer
            }
        }

        let alphaErodeRadiusPixels = state.destinationPixelRadius(fromNormalized: state.alphaErodeNormalized)
        if abs(alphaErodeRadiusPixels) > 0.5 {
            try runMorphology(
                source: current,
                intermediate: auxiliary,
                destination: buffer,
                radius: Int(alphaErodeRadiusPixels.rounded()),
                entry: entry,
                commandBuffer: commandBuffer
            )
            current = buffer
        }

        let softnessRadiusPixels = state.destinationPixelRadius(fromNormalized: state.alphaSoftnessNormalized)
        if softnessRadiusPixels > 0.5 {
            try runGaussianBlur(
                source: current,
                intermediate: auxiliary,
                destination: buffer,
                radiusPixels: softnessRadiusPixels,
                entry: entry,
                commandBuffer: commandBuffer
            )
            current = buffer
        }
        return current
    }

    private func runAlphaLevelsGamma(
        source: any MTLTexture,
        destination: any MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Levels + Gamma"
        encoder.setComputePipelineState(entry.computePipelines.alphaLevelsGamma)
        encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKAlphaEdgeParams(
            blackPoint: Float(state.alphaBlackPoint),
            whitePoint: Float(state.alphaWhitePoint),
            gamma: Float(state.alphaGamma),
            morphRadius: 0,
            blurRadius: 0
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKAlphaEdgeParams>.size,
            index: Int(CKBufferIndexAlphaEdgeParams.rawValue)
        )
        dispatch(encoder: encoder, pipeline: entry.computePipelines.alphaLevelsGamma, width: destination.width, height: destination.height)
        encoder.endEncoding()
    }

    private func runMorphology(
        source: any MTLTexture,
        intermediate: any MTLTexture,
        destination: any MTLTexture,
        radius: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        var absoluteRadius = Int32(abs(radius))
        var erodeFlag: Int32 = (radius < 0) ? 1 : 0

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Morphology H"
            encoder.setComputePipelineState(entry.computePipelines.morphologyHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&absoluteRadius, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            dispatch(encoder: encoder, pipeline: entry.computePipelines.morphologyHorizontal, width: intermediate.width, height: intermediate.height)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Morphology V"
            encoder.setComputePipelineState(entry.computePipelines.morphologyVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&absoluteRadius, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            dispatch(encoder: encoder, pipeline: entry.computePipelines.morphologyVertical, width: destination.width, height: destination.height)
            encoder.endEncoding()
        }
    }

    private func runGaussianBlur(
        source: any MTLTexture,
        intermediate: any MTLTexture,
        destination: any MTLTexture,
        radiusPixels: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let kernelRadius = Int(ceil(radiusPixels))
        guard kernelRadius > 0 else { return }

        let sigma = max(radiusPixels * 0.5, 0.5)
        var weights: [Float] = []
        weights.reserveCapacity(kernelRadius + 1)
        var total: Float = 0
        for index in 0...kernelRadius {
            let offset = Float(index)
            let weight = exp(-(offset * offset) / (2 * sigma * sigma))
            weights.append(weight)
            total += (index == 0) ? weight : (weight * 2)
        }
        for index in weights.indices { weights[index] /= total }

        guard let weightsBuffer = entry.device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else { return }

        var radiusValue = Int32(kernelRadius)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Blur H"
            encoder.setComputePipelineState(entry.computePipelines.gaussianHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weightsBuffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            dispatch(encoder: encoder, pipeline: entry.computePipelines.gaussianHorizontal, width: intermediate.width, height: intermediate.height)
            encoder.endEncoding()
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Blur V"
            encoder.setComputePipelineState(entry.computePipelines.gaussianVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weightsBuffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            dispatch(encoder: encoder, pipeline: entry.computePipelines.gaussianVertical, width: destination.width, height: destination.height)
            encoder.endEncoding()
        }
    }

    private func sourcePassthrough(
        foreground: any MTLTexture,
        source: any MTLTexture,
        matte: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        guard let output = entry.makeIntermediateTexture(width: foreground.width, height: foreground.height) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Source Passthrough"
        encoder.setComputePipelineState(entry.computePipelines.sourcePassthrough)
        encoder.setTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(output, index: Int(CKTextureIndexOutput.rawValue))
        dispatch(encoder: encoder, pipeline: entry.computePipelines.sourcePassthrough, width: output.width, height: output.height)
        encoder.endEncoding()
        return output
    }

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

        let outputWidth = Float(destinationTile.tilePixelBounds.right - destinationTile.tilePixelBounds.left)
        let outputHeight = Float(destinationTile.tilePixelBounds.top - destinationTile.tilePixelBounds.bottom)
        let halfW = outputWidth * 0.5
        let halfH = outputHeight * 0.5

        var vertices: [CKVertex2D] = [
            CKVertex2D(position: SIMD2<Float>(halfW, -halfH), textureCoordinate: SIMD2<Float>(1, 1)),
            CKVertex2D(position: SIMD2<Float>(-halfW, -halfH), textureCoordinate: SIMD2<Float>(0, 1)),
            CKVertex2D(position: SIMD2<Float>(halfW, halfH), textureCoordinate: SIMD2<Float>(1, 0)),
            CKVertex2D(position: SIMD2<Float>(-halfW, halfH), textureCoordinate: SIMD2<Float>(0, 0))
        ]
        var viewportSize = SIMD2<UInt32>(UInt32(outputWidth), UInt32(outputHeight))

        encoder.setViewport(MTLViewport(
            originX: 0,
            originY: 0,
            width: Double(outputWidth),
            height: Double(outputHeight),
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
            outputMode: state.outputMode.shaderValue
        )
        encoder.setFragmentBytes(
            &params,
            length: MemoryLayout<CKComposeParams>.size,
            index: Int(CKBufferIndexComposeParams.rawValue)
        )
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
    }

    private func resample(
        source: any MTLTexture,
        targetWidth: Int,
        targetHeight: Int,
        pixelFormat: MTLPixelFormat,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        if source.width == targetWidth && source.height == targetHeight && source.pixelFormat == pixelFormat {
            return source
        }
        guard let target = entry.makeIntermediateTexture(
            width: targetWidth,
            height: targetHeight,
            pixelFormat: pixelFormat
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Resample"
        encoder.setComputePipelineState(entry.computePipelines.resample)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(target, index: Int(CKTextureIndexOutput.rawValue))
        dispatch(encoder: encoder, pipeline: entry.computePipelines.resample, width: target.width, height: target.height)
        encoder.endEncoding()
        return target
    }

    // MARK: - Utilities

    private func dispatch(
        encoder: any MTLComputeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        let threadgroupWidth = min(pipeline.threadExecutionWidth, max(width, 1))
        let threadgroupHeight = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / max(threadgroupWidth, 1), max(height, 1)))
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        let threadsPerGrid = MTLSize(width: max(width, 1), height: max(height, 1), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    /// Converts the inspector's "Despeckle Size" (approximate speckle area in
    /// 1920px-baseline pixels) to a morphology radius scaled to the current
    /// destination. The square-root mapping treats the slider as area rather
    /// than diameter — a 100px² speckle becomes a 10px structuring radius.
    private func despeckleRadiusPixels(state: PluginStateData) -> Int {
        let scale = Double(state.destinationLongEdgePixels) / max(state.longEdgeBaseline, 1.0)
        let radius = Double(max(state.despeckleSize, 1)).squareRoot() * scale
        return max(1, Int(radius.rounded()))
    }

    private func hintTileLayoutValue(for texture: any MTLTexture) -> Int32 {
        switch texture.pixelFormat {
        case .rgba16Float, .rgba32Float, .bgra8Unorm, .rgba8Unorm: return 0
        case .r8Unorm, .r16Float, .r32Float: return 1
        default: return 2
        }
    }
}
