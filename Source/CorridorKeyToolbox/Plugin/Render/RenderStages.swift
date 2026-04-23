//
//  RenderStages.swift
//  Corridor Key Toolbox
//
//  Pure-Metal stage helpers. None of these functions reference FxPlug types —
//  they take textures, state, and a command buffer, and encode one stage of
//  the Corridor Key pipeline. This lets the unit tests drive each stage
//  directly without launching Final Cut Pro.
//
//  The `RenderPipeline` FxPlug orchestrator composes these stages into the
//  full per-frame pipeline; when we add a new shader, we add a matching
//  entry point here so both production and tests exercise the same code
//  path.
//

import Foundation
import Metal
import MetalPerformanceShaders
import simd
#if CORRIDOR_KEY_SPM_MIRROR
import CorridorKeyToolboxLogic
#endif

enum RenderStages {

    // MARK: - Screen colour matrix

    /// Applies a 3x3 colour matrix to `source`, writing into a pooled
    /// destination texture. Skips the dispatch entirely when the matrix is
    /// the identity.
    static func applyScreenMatrix(
        source: any MTLTexture,
        matrix: simd_float3x3,
        isIdentity: Bool,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        if isIdentity { return nil }
        guard let output = entry.texturePool.acquire(
            width: source.width,
            height: source.height,
            pixelFormat: .rgba16Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Screen Matrix"
        encoder.setComputePipelineState(entry.computePipelines.applyScreenMatrix)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var matrixCopy = matrix
        encoder.setBytes(
            &matrixCopy,
            length: MemoryLayout<simd_float3x3>.size,
            index: Int(CKBufferIndexScreenColorMatrix.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.applyScreenMatrix,
            width: output.texture.width,
            height: output.texture.height
        )
        encoder.endEncoding()
        return output
    }

    // MARK: - Green hint / rough matte

    /// Generates a green-bias hint texture from the source RGB.
    static func generateGreenHint(
        source: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture {
        guard let output = entry.texturePool.acquire(
            width: source.width,
            height: source.height,
            pixelFormat: .r16Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Green Hint"
        encoder.setComputePipelineState(entry.computePipelines.greenHint)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.greenHint,
            width: output.texture.width,
            height: output.texture.height
        )
        encoder.endEncoding()
        return output
    }

    /// Extracts an alpha hint out of an externally-supplied texture. The
    /// layout flag tells the kernel which channel holds the hint value
    /// (`0` = alpha for RGBA layouts, `1` = R for single-channel, `2` = R
    /// for RGB-only layouts).
    static func extractHint(
        source: any MTLTexture,
        layout: Int32,
        targetWidth: Int,
        targetHeight: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture {
        guard let output = entry.texturePool.acquire(
            width: targetWidth,
            height: targetHeight,
            pixelFormat: .r16Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Hint"
        encoder.setComputePipelineState(entry.computePipelines.extractHint)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var layoutCopy = layout
        encoder.setBytes(&layoutCopy, length: MemoryLayout<Int32>.size, index: 0)
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.extractHint,
            width: output.texture.width,
            height: output.texture.height
        )
        encoder.endEncoding()
        return output
    }

    // MARK: - Combine + normalise (pre-inference)

    /// Downsamples source + hint to `inferenceResolution` and packs into the
    /// four-channel ImageNet-normalised tensor MLX expects. `workingToRec709`
    /// converts the host's working gamut to Rec.709-linear before
    /// normalisation so HDR Rec.2020 projects predict as accurately as SDR.
    static func combineAndNormalise(
        source: any MTLTexture,
        hint: any MTLTexture,
        inferenceResolution: Int,
        workingToRec709: simd_float3x3,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture {
        guard let normalised = entry.texturePool.acquire(
            width: inferenceResolution,
            height: inferenceResolution,
            pixelFormat: .rgba32Float,
            storageMode: .shared
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            normalised.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Combine + Normalise"
        encoder.setComputePipelineState(entry.computePipelines.combineAndNormalize)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(hint, index: Int(CKTextureIndexHint.rawValue))
        encoder.setTexture(normalised.texture, index: Int(CKTextureIndexOutput.rawValue))

        var params = CKNormalizeParams(
            workingToRec709: workingToRec709,
            mean: SIMD3<Float>(0.485, 0.456, 0.406),
            invStdDev: SIMD3<Float>(1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKNormalizeParams>.size,
            index: Int(CKBufferIndexNormalizeParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.combineAndNormalize,
            width: inferenceResolution,
            height: inferenceResolution
        )
        encoder.endEncoding()
        return normalised
    }

    // MARK: - Resample (bilinear + Lanczos)

    /// Resamples to the given dimensions using either a bilinear compute
    /// kernel (fast) or MPS Lanczos (higher quality). If the input already
    /// matches the target dims and format, returns `nil` — callers continue
    /// using the original texture in that case.
    static func resample(
        source: any MTLTexture,
        targetWidth: Int,
        targetHeight: Int,
        pixelFormat: MTLPixelFormat,
        method: UpscaleMethod,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        if source.width == targetWidth && source.height == targetHeight && source.pixelFormat == pixelFormat {
            return nil
        }
        guard let target = entry.texturePool.acquire(
            width: targetWidth,
            height: targetHeight,
            pixelFormat: pixelFormat
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }

        switch method {
        case .lanczos:
            MatteRefiner.applyLanczosResample(
                source: source,
                destination: target.texture,
                entry: entry,
                commandBuffer: commandBuffer
            )
            return target

        case .bilinear:
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                target.returnManually()
                throw MetalDeviceCacheError.commandEncoderCreationFailed
            }
            encoder.label = "Corridor Key Toolbox Resample (Bilinear)"
            encoder.setComputePipelineState(entry.computePipelines.resample)
            encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
            encoder.setTexture(target.texture, index: Int(CKTextureIndexOutput.rawValue))
            dispatch(
                encoder: encoder,
                pipeline: entry.computePipelines.resample,
                width: target.texture.width,
                height: target.texture.height
            )
            encoder.endEncoding()
            return target
        }
    }

    // MARK: - Despill

    /// Runs the despill pass. `strength == 0` returns `nil` so the caller
    /// continues with the original foreground texture.
    static func despill(
        foreground: any MTLTexture,
        strength: Float,
        method: SpillMethod,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        guard strength > 0 else { return nil }
        guard let output = entry.texturePool.acquire(
            width: foreground.width,
            height: foreground.height
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Despill"
        encoder.setComputePipelineState(entry.computePipelines.despill)
        encoder.setTexture(foreground, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKDespillParams(
            strength: strength,
            method: method.shaderValue
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKDespillParams>.size,
            index: Int(CKBufferIndexDespillParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.despill,
            width: output.texture.width,
            height: output.texture.height
        )
        encoder.endEncoding()
        return output
    }

    // MARK: - Alpha levels + gamma

    static func applyAlphaLevelsGamma(
        source: any MTLTexture,
        destination: any MTLTexture,
        blackPoint: Float,
        whitePoint: Float,
        gamma: Float,
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
            blackPoint: blackPoint,
            whitePoint: whitePoint,
            gamma: gamma,
            morphRadius: 0,
            blurRadius: 0
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKAlphaEdgeParams>.size,
            index: Int(CKBufferIndexAlphaEdgeParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.alphaLevelsGamma,
            width: destination.width,
            height: destination.height
        )
        encoder.endEncoding()
    }

    // MARK: - Source passthrough

    static func sourcePassthrough(
        foreground: any MTLTexture,
        source: any MTLTexture,
        matte: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture {
        guard let output = entry.texturePool.acquire(
            width: foreground.width,
            height: foreground.height
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Source Passthrough"
        encoder.setComputePipelineState(entry.computePipelines.sourcePassthrough)
        encoder.setTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.sourcePassthrough,
            width: output.texture.width,
            height: output.texture.height
        )
        encoder.endEncoding()
        return output
    }

    // MARK: - Phase 4.1: Refiner-strength blend

    /// Blends a Gaussian-blurred "coarse" copy of `matte` with the refined
    /// matte according to `strength`. `strength == 1.0` returns `nil` —
    /// callers should just use the refined matte in that case.
    /// `strength == 0.0` writes the blurred matte out straight, producing
    /// a maximally soft pass.
    static func applyRefinerStrength(
        matte: any MTLTexture,
        strength: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        if abs(strength - 1.0) < 1e-3 { return nil }

        // Derive the coarse stand-in by Gaussian-blurring the refined matte.
        // Sigma is tuned to approximate the loss of detail the model's
        // low-resolution decoder would produce before the refiner CNN.
        let width = matte.width
        let height = matte.height
        guard let coarse = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: matte.pixelFormat
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let intermediate = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: matte.pixelFormat
        ) else {
            coarse.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        // Use a fixed 3 px sigma — tuned empirically to match the neural
        // refiner's characteristic softening radius on the bundled bridges.
        try MatteRefiner.applyGaussianBlur(
            source: matte,
            intermediate: intermediate.texture,
            destination: coarse.texture,
            radiusPixels: 3.0,
            entry: entry,
            commandBuffer: commandBuffer
        )
        intermediate.returnOnCompletion(of: commandBuffer)

        guard let blended = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: matte.pixelFormat
        ) else {
            coarse.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            coarse.returnManually()
            blended.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Refiner Blend"
        encoder.setComputePipelineState(entry.computePipelines.refinerBlend)
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(coarse.texture, index: Int(CKTextureIndexCoarse.rawValue))
        encoder.setTexture(blended.texture, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKRefinerParams(strength: strength)
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKRefinerParams>.size,
            index: Int(CKBufferIndexRefinerParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.refinerBlend,
            width: width,
            height: height
        )
        encoder.endEncoding()
        coarse.returnOnCompletion(of: commandBuffer)
        return blended
    }

    // MARK: - Phase 4.2: Connected-components despeckle

    /// Runs a GPU-parallel connected-components labelling pass on the matte
    /// and zeros any component whose pixel count is below `areaThreshold`.
    /// All work stays on the caller's single command buffer — pixel
    /// counting happens on-GPU via atomic increments into a shared buffer,
    /// so there is no CPU readback/wait in the middle of the pipeline.
    ///
    /// Returns `nil` when `areaThreshold <= 0` or when the matte is larger
    /// than the 4096² limit the label encoding supports (covers every
    /// supported FCP output frame).
    static func applyConnectedComponentsDespeckle(
        matte: any MTLTexture,
        areaThreshold: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        guard areaThreshold > 0 else { return nil }
        let width = matte.width
        let height = matte.height
        guard width <= 4096, height <= 4096 else { return nil }

        // Ping-pong label textures. r32Float stores labels exactly for the
        // values we need (integers ≤ 16.7M, well above 4096 × 4096).
        guard let labelA = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: .r32Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let labelB = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: .r32Float
        ) else {
            labelA.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }

        // `labelSpan` is used by the filter kernel as a raw bounds check
        // on the counts buffer, so it must equal `width * height + 1`
        // (labels are 1-indexed by the init pass).
        var params = CKCCLabelParams(
            areaThreshold: Int32(areaThreshold),
            labelSpan: Int32(width * height + 1),
            matteThreshold: 0.5,
            blurSigma: 0.0
        )

        // Init pass: binarise matte → unique integer label per pixel.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox CC Init"
            encoder.setComputePipelineState(entry.computePipelines.ccLabelInit)
            encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(labelA.texture, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(
                &params,
                length: MemoryLayout<CKCCLabelParams>.size,
                index: Int(CKBufferIndexCCLabelParams.rawValue)
            )
            dispatch(
                encoder: encoder,
                pipeline: entry.computePipelines.ccLabelInit,
                width: width,
                height: height
            )
            encoder.endEncoding()
        }

        // Propagate: min-neighbour flood with 1-pixel stride. Each
        // iteration extends a component's labels by one pixel, so a
        // component of diameter D converges after D iterations. We cap
        // at 256 iterations — enough for mattes up to ≈ 256 px across
        // and a reasonable upper bound for GPU work even on 4K targets.
        // Larger components with long connective tissue may not fully
        // merge, but their size almost always exceeds the despeckle
        // area threshold so they survive regardless.
        let maxDim = max(width, height)
        let iterationCount = min(256, max(32, maxDim))
        var sourceLabel = labelA
        var destinationLabel = labelB
        for _ in 0..<iterationCount {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Corridor Key Toolbox CC Propagate"
                encoder.setComputePipelineState(entry.computePipelines.ccLabelPropagate)
                encoder.setTexture(sourceLabel.texture, index: Int(CKTextureIndexMatte.rawValue))
                encoder.setTexture(destinationLabel.texture, index: Int(CKTextureIndexOutput.rawValue))
                var stride: Int32 = 1
                encoder.setBytes(&stride, length: MemoryLayout<Int32>.size, index: 0)
                dispatch(
                    encoder: encoder,
                    pipeline: entry.computePipelines.ccLabelPropagate,
                    width: width,
                    height: height
                )
                encoder.endEncoding()
            }
            swap(&sourceLabel, &destinationLabel)
        }

        // Atomic counts buffer — zeroed, filled by the count kernel, read
        // by the filter kernel. Pixel count + 2 entries so labels assigned
        // at the bottom-right corner (`width * height`) have a valid slot,
        // plus index 0 stays reserved for background.
        let labelCapacity = width * height + 2
        guard let countsBuffer = entry.device.makeBuffer(
            length: labelCapacity * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            labelA.returnManually()
            labelB.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        countsBuffer.label = "Corridor Key Toolbox CC Counts"
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "Corridor Key Toolbox CC Counts Zero"
            blit.fill(buffer: countsBuffer, range: 0..<countsBuffer.length, value: 0)
            blit.endEncoding()
        }

        // Atomic count pass.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox CC Count"
            encoder.setComputePipelineState(entry.computePipelines.ccLabelCount)
            encoder.setTexture(sourceLabel.texture, index: Int(CKTextureIndexLabel.rawValue))
            encoder.setBuffer(countsBuffer, offset: 0, index: Int(CKBufferIndexCCLabelCounts.rawValue))
            encoder.setBytes(
                &params,
                length: MemoryLayout<CKCCLabelParams>.size,
                index: Int(CKBufferIndexCCLabelParams.rawValue)
            )
            dispatch(
                encoder: encoder,
                pipeline: entry.computePipelines.ccLabelCount,
                width: width,
                height: height
            )
            encoder.endEncoding()
        }

        // Filter pass: zero any pixel whose component count is below
        // threshold. Writes into a fresh pooled matte texture.
        guard let filtered = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: matte.pixelFormat
        ) else {
            labelA.returnManually()
            labelB.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox CC Filter"
            encoder.setComputePipelineState(entry.computePipelines.ccLabelFilter)
            encoder.setTexture(sourceLabel.texture, index: Int(CKTextureIndexLabel.rawValue))
            encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(filtered.texture, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(countsBuffer, offset: 0, index: Int(CKBufferIndexCCLabelCounts.rawValue))
            encoder.setBytes(
                &params,
                length: MemoryLayout<CKCCLabelParams>.size,
                index: Int(CKBufferIndexCCLabelParams.rawValue)
            )
            dispatch(
                encoder: encoder,
                pipeline: entry.computePipelines.ccLabelFilter,
                width: width,
                height: height
            )
            encoder.endEncoding()
        }

        labelA.returnOnCompletion(of: commandBuffer)
        labelB.returnOnCompletion(of: commandBuffer)
        return filtered
    }

    // MARK: - Phase 4.3: Light wrap

    static func applyLightWrap(
        foreground: any MTLTexture,
        matte: any MTLTexture,
        sourceRGB: any MTLTexture,
        radiusPixels: Float,
        strength: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        guard strength > 0, radiusPixels > 0 else { return nil }

        // Blur a copy of the source RGB so we sample ambient colour rather
        // than high-frequency noise. Uses the same MPS path as the matte
        // softness blur — shares the cached `MPSImageGaussianBlur` instance
        // when possible.
        let width = foreground.width
        let height = foreground.height
        guard let blurredSource = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: sourceRGB.pixelFormat
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let intermediate = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: sourceRGB.pixelFormat
        ) else {
            blurredSource.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        try MatteRefiner.applyGaussianBlur(
            source: sourceRGB,
            intermediate: intermediate.texture,
            destination: blurredSource.texture,
            radiusPixels: radiusPixels,
            entry: entry,
            commandBuffer: commandBuffer
        )
        intermediate.returnOnCompletion(of: commandBuffer)

        guard let output = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: foreground.pixelFormat
        ) else {
            blurredSource.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            blurredSource.returnManually()
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Light Wrap"
        encoder.setComputePipelineState(entry.computePipelines.lightWrap)
        encoder.setTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setTexture(blurredSource.texture, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKLightWrapParams(strength: strength, edgeBias: 0.6)
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKLightWrapParams>.size,
            index: Int(CKBufferIndexLightWrapParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.lightWrap,
            width: width,
            height: height
        )
        encoder.endEncoding()
        blurredSource.returnOnCompletion(of: commandBuffer)
        return output
    }

    // MARK: - Phase 4.4: Edge decontamination

    static func applyEdgeDecontamination(
        foreground: any MTLTexture,
        matte: any MTLTexture,
        screenColor: SIMD3<Float>,
        strength: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        guard strength > 0 else { return nil }

        let width = foreground.width
        let height = foreground.height
        guard let output = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: foreground.pixelFormat
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "Corridor Key Toolbox Edge Decontaminate"
        encoder.setComputePipelineState(entry.computePipelines.edgeDecontaminate)
        encoder.setTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKEdgeDecontaminateParams(
            strength: strength,
            screenColor: screenColor
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKEdgeDecontaminateParams>.size,
            index: Int(CKBufferIndexEdgeDecontaminateParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.edgeDecontaminate,
            width: width,
            height: height
        )
        encoder.endEncoding()
        return output
    }

    // MARK: - Threadgroup dispatch helper

    static func dispatch(
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
}
