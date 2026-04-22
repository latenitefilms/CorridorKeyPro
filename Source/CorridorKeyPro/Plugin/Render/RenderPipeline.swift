//
//  RenderPipeline.swift
//  Corridor Key Pro
//
//  Orchestrates a single-frame render: screen-colour mapping, downsample,
//  normalisation, neural inference, despill, alpha edge work, source
//  passthrough blend, and final compose. Every stage runs on the GPU; the host
//  hands us a destination `IOSurface` and we write straight back to it.
//

import Foundation
import Metal
import CoreMedia
import simd

/// Input bundle for a render. Kept as a value type so that the orchestrator is
/// trivial to reason about in isolation.
struct RenderRequest: @unchecked Sendable {
    let destinationImage: FxImageTile
    let sourceImage: FxImageTile
    let alphaHintImage: FxImageTile?
    let state: PluginStateData
    let renderTime: CMTime
}

/// Runs the per-frame pipeline. One instance is created per FxPlug plug-in
/// instance so each timeline can hold its own warmed-up model without stepping
/// on its neighbours. Final Cut Pro may invoke `render(_:)` concurrently for
/// multiple tiles, so the pipeline keeps no per-frame mutable state of its own
/// – each call is self-contained and the shared `InferenceCoordinator` is
/// internally locked.
final class RenderPipeline: @unchecked Sendable {
    private let deviceCache: MetalDeviceCache
    private let inferenceCoordinator: InferenceCoordinator

    init(
        deviceCache: MetalDeviceCache = .shared,
        inferenceCoordinator: InferenceCoordinator = InferenceCoordinator()
    ) {
        self.deviceCache = deviceCache
        self.inferenceCoordinator = inferenceCoordinator
    }

    /// Executes the full render for one tile. Throws if Metal resources can't
    /// be provisioned; the plug-in surfaces a user-visible error in that case.
    ///
    /// The render is split into three phases so the inference engine can
    /// read from and write to `.shared` textures without racing GPU work:
    ///
    /// 1. Pre-inference: rotate, resample, combine + normalise — committed
    ///    and awaited so the normalised input texture is populated.
    /// 2. Inference: the engine runs synchronously, reads the shared input
    ///    and writes the shared alpha/foreground outputs.
    /// 3. Post-inference: upscale, despill, refine matte, passthrough,
    ///    restore, compose — committed and awaited.
    func render(_ request: RenderRequest) throws {
        let destinationTile = request.destinationImage
        let sourceTile = request.sourceImage

        let pixelFormat = MetalDeviceCache.metalPixelFormat(for: destinationTile)
        guard let device = deviceCache.device(forRegistryID: destinationTile.deviceRegistryID) else {
            throw MetalDeviceCacheError.unknownDevice(destinationTile.deviceRegistryID)
        }
        let entry = try deviceCache.entry(for: device, pixelFormat: pixelFormat)

        guard let commandQueue = entry.borrowCommandQueue() else {
            throw MetalDeviceCacheError.queueExhausted
        }
        defer { entry.returnCommandQueue(commandQueue) }

        guard let sourceTexture = sourceTile.metalTexture(for: device),
              let destinationTexture = destinationTile.metalTexture(for: device) else {
            throw MetalDeviceCacheError.unknownDevice(destinationTile.deviceRegistryID)
        }

        let destinationWidth = sourceTexture.width
        let destinationHeight = sourceTexture.height

        // Pre-inference phase -------------------------------------------------
        guard let preCommandBuffer = commandQueue.makeCommandBuffer() else { return }
        preCommandBuffer.label = "Corridor Key Pro Pre-Inference"

        let screenTransform = ScreenColorEstimator.defaultTransform(for: request.state.screenColor)

        let workingSource = try rotateIntoGreenDomain(
            source: sourceTexture,
            transform: screenTransform,
            entry: entry,
            commandBuffer: preCommandBuffer
        )

        let hintTexture = try makeHintTexture(
            source: workingSource,
            hintTile: request.alphaHintImage,
            device: device,
            entry: entry,
            commandBuffer: preCommandBuffer
        )

        let inferenceResolution = request.state.qualityMode.resolvedInferenceResolution(
            forLongEdge: max(destinationWidth, destinationHeight)
        )
        let normalisedInput = try combineAndNormalise(
            source: workingSource,
            hint: hintTexture,
            inferenceResolution: inferenceResolution,
            entry: entry,
            commandBuffer: preCommandBuffer
        )

        preCommandBuffer.commit()
        preCommandBuffer.waitUntilCompleted()
        if let error = preCommandBuffer.error { throw error }

        // Inference phase -----------------------------------------------------
        let inferenceOutput = try inferenceCoordinator.runInference(
            request: KeyingInferenceRequest(
                normalisedInputTexture: normalisedInput,
                inferenceResolution: inferenceResolution
            ),
            cacheEntry: entry
        )

        // Post-inference phase ------------------------------------------------
        guard let postCommandBuffer = commandQueue.makeCommandBuffer() else { return }
        postCommandBuffer.label = "Corridor Key Pro Post-Inference"

        let upscaledAlpha = try resample(
            source: inferenceOutput.alphaTexture,
            targetWidth: destinationWidth,
            targetHeight: destinationHeight,
            entry: entry,
            commandBuffer: postCommandBuffer
        )
        let upscaledForeground = try resample(
            source: inferenceOutput.foregroundTexture,
            targetWidth: destinationWidth,
            targetHeight: destinationHeight,
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
                source: workingSource,
                matte: refinedMatte,
                state: request.state,
                entry: entry,
                commandBuffer: postCommandBuffer
            )
        } else {
            workingForeground = despilled
        }

        let restoredForeground = try restoreOriginalDomain(
            foreground: workingForeground,
            transform: screenTransform,
            entry: entry,
            commandBuffer: postCommandBuffer
        )

        try compose(
            foreground: restoredForeground,
            source: sourceTexture,
            matte: refinedMatte,
            destination: destinationTexture,
            state: request.state,
            entry: entry,
            commandBuffer: postCommandBuffer
        )

        postCommandBuffer.commit()
        postCommandBuffer.waitUntilCompleted()
        if let error = postCommandBuffer.error { throw error }
    }

    // MARK: - Stage helpers

    private func rotateIntoGreenDomain(
        source: MTLTexture,
        transform: ScreenColorTransform,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard !transform.isIdentity else { return source }
        guard let output = entry.makeIntermediateTexture(width: source.width, height: source.height) else {
            return source
        }
        let passDescriptor = MTLRenderPassDescriptor()
        passDescriptor.colorAttachments[0].texture = output
        passDescriptor.colorAttachments[0].loadAction = .dontCare
        passDescriptor.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
            return source
        }
        encoder.label = "Corridor Key Screen Rotation"
        encoder.setRenderPipelineState(entry.pipelines.applyColorMatrix)
        setupFullscreenQuad(on: encoder, width: output.width, height: output.height)
        encoder.setFragmentTexture(source, index: Int(CKTextureIndexSource.rawValue))
        var matrix = transform.forwardMatrix
        encoder.setFragmentBytes(&matrix, length: MemoryLayout<simd_float3x3>.size, index: Int(CKBufferIndexScreenColorMatrix.rawValue))
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        return output
    }

    private func restoreOriginalDomain(
        foreground: MTLTexture,
        transform: ScreenColorTransform,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard !transform.isIdentity else { return foreground }
        guard let output = entry.makeIntermediateTexture(width: foreground.width, height: foreground.height) else {
            return foreground
        }
        let passDescriptor = MTLRenderPassDescriptor()
        passDescriptor.colorAttachments[0].texture = output
        passDescriptor.colorAttachments[0].loadAction = .dontCare
        passDescriptor.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
            return foreground
        }
        encoder.label = "Corridor Key Screen Restore"
        encoder.setRenderPipelineState(entry.pipelines.applyColorMatrix)
        setupFullscreenQuad(on: encoder, width: output.width, height: output.height)
        encoder.setFragmentTexture(foreground, index: Int(CKTextureIndexSource.rawValue))
        var matrix = transform.inverseMatrix
        encoder.setFragmentBytes(&matrix, length: MemoryLayout<simd_float3x3>.size, index: Int(CKBufferIndexScreenColorMatrix.rawValue))
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        return output
    }

    private func makeHintTexture(
        source: MTLTexture,
        hintTile: FxImageTile?,
        device: MTLDevice,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard let hintTexture = entry.makeIntermediateTexture(
            width: source.width,
            height: source.height,
            pixelFormat: .r16Float
        ) else {
            throw MetalDeviceCacheError.missingDefaultLibrary
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return hintTexture }
        encoder.label = "Corridor Key Hint"

        if let hintTile, let hostTexture = hintTile.metalTexture(for: device) {
            encoder.setComputePipelineState(entry.pipelines.extractHint)
            encoder.setTexture(hostTexture, index: Int(CKTextureIndexSource.rawValue))
            encoder.setTexture(hintTexture, index: Int(CKTextureIndexOutput.rawValue))
            var layout: Int32 = hintTileLayoutValue(for: hostTexture)
            encoder.setBytes(&layout, length: MemoryLayout<Int32>.size, index: 0)
        } else {
            encoder.setComputePipelineState(entry.pipelines.roughMatte)
            encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
            encoder.setTexture(hintTexture, index: Int(CKTextureIndexOutput.rawValue))
        }

        let threads = MTLSize(width: hintTexture.width, height: hintTexture.height, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.roughMatte, threads: threads)
        )
        encoder.endEncoding()
        return hintTexture
    }

    /// Downsamples source + hint to the inference resolution and writes the
    /// four-channel normalised tensor the neural model expects into a single
    /// `.shared` texture. `.shared` storage is required so the inference
    /// engine can read the result back from the CPU.
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
            throw MetalDeviceCacheError.missingDefaultLibrary
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return normalised }
        encoder.label = "Corridor Key Combine + Normalise"
        encoder.setComputePipelineState(entry.pipelines.combineAndNormalize)
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

        let threads = MTLSize(width: inferenceResolution, height: inferenceResolution, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.combineAndNormalize, threads: threads)
        )
        encoder.endEncoding()
        return normalised
    }

    private func despill(
        foreground: MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard state.despillStrength > 0 else { return foreground }
        guard let output = entry.makeIntermediateTexture(width: foreground.width, height: foreground.height) else {
            return foreground
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return foreground }
        encoder.label = "Corridor Key Despill"
        encoder.setComputePipelineState(entry.pipelines.despill)
        encoder.setTexture(foreground, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKDespillParams(
            strength: Float(state.despillStrength),
            method: state.spillMethod.shaderValue
        )
        encoder.setBytes(&params, length: MemoryLayout<CKDespillParams>.size, index: Int(CKBufferIndexDespillParams.rawValue))
        let threads = MTLSize(width: output.width, height: output.height, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.despill, threads: threads)
        )
        encoder.endEncoding()
        return output
    }

    private func refineMatte(
        alpha: MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        // Levels and gamma run first so downstream morphology and blur work on
        // the user-corrected alpha. Each pass is a compute shader that reads
        // from `current` and writes to `next`, then swaps.
        var current = alpha
        guard let buffer = entry.makeIntermediateTexture(
            width: alpha.width,
            height: alpha.height,
            pixelFormat: .r16Float
        ) else { return alpha }
        guard let auxiliary = entry.makeIntermediateTexture(
            width: alpha.width,
            height: alpha.height,
            pixelFormat: .r16Float
        ) else { return alpha }

        // Levels + gamma
        try runAlphaLevelsGamma(source: current, destination: buffer, state: state, entry: entry, commandBuffer: commandBuffer)
        current = buffer

        // Erode / dilate when the user asked for a non-zero alpha erode.
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

        // Gaussian softness
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
        source: MTLTexture,
        destination: MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Corridor Key Levels + Gamma"
        encoder.setComputePipelineState(entry.pipelines.alphaLevelsGamma)
        encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKAlphaEdgeParams(
            blackPoint: Float(state.alphaBlackPoint),
            whitePoint: Float(state.alphaWhitePoint),
            gamma: Float(state.alphaGamma),
            morphRadius: 0,
            blurRadius: 0
        )
        encoder.setBytes(&params, length: MemoryLayout<CKAlphaEdgeParams>.size, index: Int(CKBufferIndexAlphaEdgeParams.rawValue))
        let threads = MTLSize(width: destination.width, height: destination.height, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.alphaLevelsGamma, threads: threads)
        )
        encoder.endEncoding()
    }

    private func runMorphology(
        source: MTLTexture,
        intermediate: MTLTexture,
        destination: MTLTexture,
        radius: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws {
        let absoluteRadius = Int32(abs(radius))
        var radiusBuffer = absoluteRadius
        var erodeFlag: Int32 = (radius < 0) ? 1 : 0

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Morphology H"
            encoder.setComputePipelineState(entry.pipelines.morphologyHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&radiusBuffer, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            let threads = MTLSize(width: intermediate.width, height: intermediate.height, depth: 1)
            encoder.dispatchThreads(
                threads,
                threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.morphologyHorizontal, threads: threads)
            )
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Morphology V"
            encoder.setComputePipelineState(entry.pipelines.morphologyVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&radiusBuffer, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            let threads = MTLSize(width: destination.width, height: destination.height, depth: 1)
            encoder.dispatchThreads(
                threads,
                threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.morphologyVertical, threads: threads)
            )
            encoder.endEncoding()
        }
    }

    private func runGaussianBlur(
        source: MTLTexture,
        intermediate: MTLTexture,
        destination: MTLTexture,
        radiusPixels: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws {
        let kernelRadius = Int(ceil(radiusPixels))
        guard kernelRadius > 0 else { return }

        // Build separable Gaussian weights on the CPU and upload once.
        let sigma = radiusPixels * 0.5
        var weights: [Float] = []
        weights.reserveCapacity(kernelRadius + 1)
        var total: Float = 0
        for i in 0...kernelRadius {
            let f = Float(i)
            let weight = exp(-(f * f) / (2 * sigma * sigma))
            weights.append(weight)
            total += (i == 0) ? weight : (weight * 2)
        }
        for index in weights.indices { weights[index] /= total }

        guard let weightsBuffer = entry.device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else { return }

        var radiusValue = Int32(kernelRadius)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Blur H"
            encoder.setComputePipelineState(entry.pipelines.gaussianHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weightsBuffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            let threads = MTLSize(width: intermediate.width, height: intermediate.height, depth: 1)
            encoder.dispatchThreads(
                threads,
                threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.gaussianHorizontal, threads: threads)
            )
            encoder.endEncoding()
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Blur V"
            encoder.setComputePipelineState(entry.pipelines.gaussianVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weightsBuffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            let threads = MTLSize(width: destination.width, height: destination.height, depth: 1)
            encoder.dispatchThreads(
                threads,
                threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.gaussianVertical, threads: threads)
            )
            encoder.endEncoding()
        }
    }

    private func sourcePassthrough(
        foreground: MTLTexture,
        source: MTLTexture,
        matte: MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard let output = entry.makeIntermediateTexture(width: foreground.width, height: foreground.height) else {
            return foreground
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return foreground }
        encoder.label = "Corridor Key Source Passthrough"
        encoder.setComputePipelineState(entry.pipelines.sourcePassthrough)
        encoder.setTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(output, index: Int(CKTextureIndexOutput.rawValue))
        let threads = MTLSize(width: output.width, height: output.height, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.sourcePassthrough, threads: threads)
        )
        encoder.endEncoding()
        return output
    }

    private func compose(
        foreground: MTLTexture,
        source: MTLTexture,
        matte: MTLTexture,
        destination: MTLTexture,
        state: PluginStateData,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Corridor Key Compose"
        encoder.setComputePipelineState(entry.pipelines.compose)
        encoder.setTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKComposeParams(
            outputMode: state.outputMode.shaderValue,
            temporalSmoothing: Float(state.temporalSmoothing)
        )
        encoder.setBytes(&params, length: MemoryLayout<CKComposeParams>.size, index: Int(CKBufferIndexComposeParams.rawValue))
        let threads = MTLSize(width: destination.width, height: destination.height, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.compose, threads: threads)
        )
        encoder.endEncoding()
    }

    private func resample(
        source: MTLTexture,
        targetWidth: Int,
        targetHeight: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        if source.width == targetWidth && source.height == targetHeight { return source }
        guard let target = entry.makeIntermediateTexture(
            width: targetWidth,
            height: targetHeight,
            pixelFormat: source.pixelFormat
        ) else { return source }
        try resampleInPlace(source: source, target: target, entry: entry, commandBuffer: commandBuffer)
        return target
    }

    private func resampleInPlace(
        source: MTLTexture,
        target: MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Corridor Key Resample"
        encoder.setComputePipelineState(entry.pipelines.resample)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(target, index: Int(CKTextureIndexOutput.rawValue))
        let threads = MTLSize(width: target.width, height: target.height, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: threadgroupSize(for: entry.pipelines.resample, threads: threads)
        )
        encoder.endEncoding()
    }

    // MARK: - Utilities

    private func setupFullscreenQuad(on encoder: MTLRenderCommandEncoder, width: Int, height: Int) {
        let halfWidth = Float(width) * 0.5
        let halfHeight = Float(height) * 0.5
        var vertices: [CKVertex2D] = [
            CKVertex2D(position: SIMD2<Float>(halfWidth, -halfHeight), textureCoordinate: SIMD2<Float>(1, 1)),
            CKVertex2D(position: SIMD2<Float>(-halfWidth, -halfHeight), textureCoordinate: SIMD2<Float>(0, 1)),
            CKVertex2D(position: SIMD2<Float>(halfWidth, halfHeight), textureCoordinate: SIMD2<Float>(1, 0)),
            CKVertex2D(position: SIMD2<Float>(-halfWidth, halfHeight), textureCoordinate: SIMD2<Float>(0, 0))
        ]
        encoder.setVertexBytes(&vertices, length: MemoryLayout<CKVertex2D>.stride * vertices.count, index: Int(CKVertexInputIndexVertices.rawValue))
        var viewport = SIMD2<UInt32>(UInt32(width), UInt32(height))
        encoder.setVertexBytes(&viewport, length: MemoryLayout<SIMD2<UInt32>>.size, index: Int(CKVertexInputIndexViewportSize.rawValue))
        encoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: -1, zfar: 1))
    }

    private func threadgroupSize(for pipeline: MTLComputePipelineState, threads: MTLSize) -> MTLSize {
        let width = min(pipeline.threadExecutionWidth, threads.width)
        let height = min(pipeline.maxTotalThreadsPerThreadgroup / max(width, 1), threads.height)
        return MTLSize(width: max(width, 1), height: max(height, 1), depth: 1)
    }

    private func hintTileLayoutValue(for texture: MTLTexture) -> Int32 {
        // 0 = RGBA (use alpha), 1 = Alpha only, 2 = RGB (use red channel).
        switch texture.pixelFormat {
        case .rgba16Float, .rgba32Float, .bgra8Unorm, .rgba8Unorm: return 0
        case .r8Unorm, .r16Float, .r32Float: return 1
        default: return 2
        }
    }
}
