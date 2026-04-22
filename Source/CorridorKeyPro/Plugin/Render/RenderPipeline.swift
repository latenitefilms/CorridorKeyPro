//
//  RenderPipeline.swift
//  Corridor Key Pro
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
        defer { entry.returnCommandQueue(commandQueue) }

        guard let sourceTexture = sourceTile.metalTexture(for: device) else {
            throw MetalDeviceCacheError.unknownDevice(destinationTile.deviceRegistryID)
        }
        guard let destinationTexture = destinationTile.metalTexture(for: device) else {
            throw MetalDeviceCacheError.unknownDevice(destinationTile.deviceRegistryID)
        }

        let sourceWidth = sourceTexture.width
        let sourceHeight = sourceTexture.height
        let longEdge = max(sourceWidth, sourceHeight)

        let inferenceResolution = request.state.qualityMode.resolvedInferenceResolution(forLongEdge: longEdge)

        // ----- Pre-inference pass -----------------------------------------
        guard let preCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        preCommandBuffer.label = "Corridor Key Pro Pre-Inference"

        let screenTransform = ScreenColorEstimator.defaultTransform(for: request.state.screenColor)

        let rotatedSource = try applyScreenRotation(
            source: sourceTexture,
            matrix: screenTransform.forwardMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: entry,
            commandBuffer: preCommandBuffer
        )

        let hintTexture = try makeHintTexture(
            source: rotatedSource,
            hintTile: request.alphaHintImage,
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
        if let error = preCommandBuffer.error {
            throw error
        }

        // ----- Inference pass ---------------------------------------------
        let inferenceOutput = try inferenceCoordinator.runInference(
            request: KeyingInferenceRequest(
                normalisedInputTexture: normalisedInput,
                rawSourceTexture: rawSourceAtInferenceResolution,
                inferenceResolution: inferenceResolution
            ),
            cacheEntry: entry
        )

        // ----- Post-inference pass ----------------------------------------
        guard let postCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        postCommandBuffer.label = "Corridor Key Pro Post-Inference"

        let upscaledAlpha = try resample(
            source: inferenceOutput.alphaTexture,
            targetWidth: sourceWidth,
            targetHeight: sourceHeight,
            pixelFormat: .r16Float,
            entry: entry,
            commandBuffer: postCommandBuffer
        )
        let upscaledForeground = try resample(
            source: inferenceOutput.foregroundTexture,
            targetWidth: sourceWidth,
            targetHeight: sourceHeight,
            pixelFormat: .rgba16Float,
            entry: entry,
            commandBuffer: postCommandBuffer
        )

        let despilled = try despill(
            foreground: upscaledForeground,
            state: request.state,
            entry: entry,
            commandBuffer: postCommandBuffer
        )

        let refinedMatte = try refineMatte(
            alpha: upscaledAlpha,
            state: request.state,
            entry: entry,
            commandBuffer: postCommandBuffer
        )

        let workingForeground: any MTLTexture
        if request.state.sourcePassthroughEnabled {
            workingForeground = try sourcePassthrough(
                foreground: despilled,
                source: rotatedSource,
                matte: refinedMatte,
                entry: entry,
                commandBuffer: postCommandBuffer
            )
        } else {
            workingForeground = despilled
        }

        let restoredForeground = try applyScreenRotation(
            source: workingForeground,
            matrix: screenTransform.inverseMatrix,
            isIdentity: screenTransform.isIdentity,
            entry: entry,
            commandBuffer: postCommandBuffer
        )

        try compose(
            source: sourceTexture,
            foreground: restoredForeground,
            matte: refinedMatte,
            destination: destinationTexture,
            destinationTile: destinationTile,
            state: request.state,
            pixelFormat: pixelFormat,
            entry: entry,
            commandBuffer: postCommandBuffer
        )

        postCommandBuffer.commit()
        postCommandBuffer.waitUntilScheduled()
        if let error = postCommandBuffer.error {
            throw error
        }

        return RenderReport(
            backendDescription: inferenceCoordinator.backendDescription,
            guideSourceDescription: request.alphaHintImage == nil ? "Auto Rough Fallback" : "Alpha Hint Clip",
            effectiveInferenceResolution: inferenceResolution,
            deviceName: device.name
        )
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
        encoder.label = "Corridor Key Pro Screen Matrix"
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
        encoder.label = "Corridor Key Pro Hint"

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
        encoder.label = "Corridor Key Pro Combine + Normalise"
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
        encoder.label = "Corridor Key Pro Despill"
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
        encoder.label = "Corridor Key Pro Levels + Gamma"
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
            encoder.label = "Corridor Key Pro Morphology H"
            encoder.setComputePipelineState(entry.computePipelines.morphologyHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&absoluteRadius, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            dispatch(encoder: encoder, pipeline: entry.computePipelines.morphologyHorizontal, width: intermediate.width, height: intermediate.height)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Pro Morphology V"
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
            encoder.label = "Corridor Key Pro Blur H"
            encoder.setComputePipelineState(entry.computePipelines.gaussianHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weightsBuffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            dispatch(encoder: encoder, pipeline: entry.computePipelines.gaussianHorizontal, width: intermediate.width, height: intermediate.height)
            encoder.endEncoding()
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Pro Blur V"
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
        encoder.label = "Corridor Key Pro Source Passthrough"
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
        encoder.label = "Corridor Key Pro Compose"

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
            outputMode: state.outputMode.shaderValue,
            temporalSmoothing: Float(state.temporalSmoothing)
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
        encoder.label = "Corridor Key Pro Resample"
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

    private func hintTileLayoutValue(for texture: any MTLTexture) -> Int32 {
        switch texture.pixelFormat {
        case .rgba16Float, .rgba32Float, .bgra8Unorm, .rgba8Unorm: return 0
        case .r8Unorm, .r16Float, .r32Float: return 1
        default: return 2
        }
    }
}
