//
//  RenderStages.swift
//  CorridorKey by LateNite
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
        encoder.label = "CorridorKey by LateNite Screen Matrix"
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

    // MARK: - Chroma-prior hint / rough matte

    /// Combines two single-channel alpha-hint textures pixel-wise via
    /// `max(a, b)` so foreground signal from either input survives. The
    /// renderer uses this when Apple Vision is the chosen hint source —
    /// Vision returns a strong subject mask but only for people /
    /// animals / salient objects, so we fold the chroma prior in
    /// underneath to recover foreground props that Vision ignored.
    /// Both inputs must be the same size and format.
    static func unionHints(
        primary: any MTLTexture,
        secondary: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture {
        guard primary.width == secondary.width, primary.height == secondary.height else {
            // Mismatched resolutions would silently key the wrong region;
            // fail loudly so the caller surfaces the mistake instead of
            // a corrupt matte.
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let output = entry.texturePool.acquire(
            width: primary.width,
            height: primary.height,
            pixelFormat: .r16Float
        ) else { throw MetalDeviceCacheError.textureAllocationFailed }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Hint Union"
        encoder.setComputePipelineState(entry.computePipelines.hintUnion)
        encoder.setTexture(primary, index: Int(CKTextureIndexHint.rawValue))
        encoder.setTexture(secondary, index: Int(CKTextureIndexCoarse.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.hintUnion,
            width: output.texture.width,
            height: output.texture.height
        )
        encoder.endEncoding()
        return output
    }

    /// Generates a screen-colour-bias hint texture from the source RGB.
    /// The kernel reads `screenColor` to pick which channel to treat as
    /// the screen — pass `ScreenColor.green.canonicalScreenReference`
    /// for green keys, `ScreenColor.blue.canonicalScreenReference` for
    /// blue keys. Now that the renderer no longer rotates blue input
    /// into the green domain, getting this colour right is critical:
    /// a green-biased hint applied to blue-screen RGB produces a hint
    /// that says "everything is foreground" and the model has nothing
    /// useful to refine.
    static func generateChromaHint(
        source: any MTLTexture,
        screenColor: SIMD3<Float>,
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
        encoder.label = "CorridorKey by LateNite Chroma Hint"
        encoder.setComputePipelineState(entry.computePipelines.chromaHint)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKChromaHintParams(screenColor: screenColor)
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKChromaHintParams>.size,
            index: Int(CKBufferIndexChromaHintParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.chromaHint,
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
        encoder.label = "CorridorKey by LateNite Hint"
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

    // MARK: - Hint-point overlay

    /// In-place applies user-placed hint points (foreground / background
    /// dots from the on-screen control) on top of an existing hint
    /// texture. The texture must be created with `.shaderRead | .shaderWrite`
    /// usage because the kernel reads-modifies-writes each pixel. Skipped
    /// entirely when `points` is empty so a render with no OSC dots
    /// pays zero cost.
    static func applyHintPoints(
        hint: any MTLTexture,
        points: [HintPoint],
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard !points.isEmpty else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Hint Points"
        encoder.setComputePipelineState(entry.computePipelines.applyHintPoints)
        encoder.setTexture(hint, index: Int(CKTextureIndexOutput.rawValue))

        // Pack the points into a contiguous SIMD-friendly byte buffer.
        // 16 bytes per point (4 × float32) is small enough to inline as
        // setBytes — typical sessions will have <50 points so we never
        // exceed the 4 KB inline limit.
        struct PackedPoint {
            var x: Float32
            var y: Float32
            var radius: Float32
            var kind: Int32
        }
        var packed: [PackedPoint] = points.map {
            PackedPoint(
                x: Float($0.x),
                y: Float($0.y),
                radius: Float($0.radiusNormalized),
                kind: Int32($0.kind.rawValue)
            )
        }
        var pointCount: Int32 = Int32(packed.count)
        packed.withUnsafeMutableBytes { rawBytes in
            if let base = rawBytes.baseAddress {
                encoder.setBytes(base, length: rawBytes.count, index: 0)
            }
        }
        encoder.setBytes(&pointCount, length: MemoryLayout<Int32>.size, index: 1)

        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.applyHintPoints,
            width: hint.width,
            height: hint.height
        )
        encoder.endEncoding()
    }

    // MARK: - Combine + normalise (pre-inference)

    /// Production variant: downsamples source + hint to
    /// `inferenceResolution` and writes the NHWC float32 tensor directly
    /// into a cached `MTLBuffer` so MLX can read it via `init(rawPointer:)`
    /// without a CPU copy. Returns the cached buffer — callers don't own
    /// it, the `MetalDeviceCacheEntry` does.
    static func combineAndNormaliseIntoBuffer(
        source: any MTLTexture,
        hint: any MTLTexture,
        inferenceResolution: Int,
        workingToRec709: simd_float3x3,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> any MTLBuffer {
        guard let buffer = entry.normalizedInputBuffer(forRung: inferenceResolution) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Normalise → Buffer"
        encoder.setComputePipelineState(entry.computePipelines.normalizeToBuffer)
        encoder.setTexture(source, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(hint, index: Int(CKTextureIndexHint.rawValue))
        encoder.setBuffer(buffer, offset: 0, index: 0)

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
        var dims = SIMD2<UInt32>(UInt32(inferenceResolution), UInt32(inferenceResolution))
        encoder.setBytes(
            &dims,
            length: MemoryLayout<SIMD2<UInt32>>.size,
            index: 1
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.normalizeToBuffer,
            width: inferenceResolution,
            height: inferenceResolution
        )
        encoder.endEncoding()
        return buffer
    }

    /// Test-only variant: writes the normalised tensor into a pooled
    /// `.rgba32Float` texture so the SPM golden tests can read it back
    /// without having to pre-build a host-visible MTLBuffer. Kept intact
    /// so any downstream code that wants the texture form can use it.
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
        encoder.label = "CorridorKey by LateNite Combine + Normalise (Texture)"
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

    /// Encodes a compute pass that pulls MLX's 1-channel alpha output
    /// buffer into the supplied `r32Float` texture, flipping y to the
    /// Metal top-left origin the compose shader expects.
    static func writeAlphaBufferToTexture(
        buffer: any MTLBuffer,
        destination: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Alpha Buffer → Texture"
        encoder.setComputePipelineState(entry.computePipelines.alphaBufferToTexture)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.alphaBufferToTexture,
            width: destination.width,
            height: destination.height
        )
        encoder.endEncoding()
    }

    /// Encodes a compute pass that pulls MLX's 3-channel foreground
    /// output buffer into the supplied `rgba32Float` texture (alpha = 1),
    /// flipping y and expanding RGB → RGBA in the same pass.
    static func writeForegroundBufferToTexture(
        buffer: any MTLBuffer,
        destination: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Foreground Buffer → Texture"
        encoder.setComputePipelineState(entry.computePipelines.foregroundBufferToTexture)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.foregroundBufferToTexture,
            width: destination.width,
            height: destination.height
        )
        encoder.endEncoding()
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
            encoder.label = "CorridorKey by LateNite Resample (Bilinear)"
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
    /// continues with the original foreground texture. `screenColor` is
    /// the canonical reference for the keyed colour — the kernel uses
    /// it to identify which channel carries the spill, so the same
    /// pipeline despills both green and blue without per-colour code.
    static func despill(
        foreground: any MTLTexture,
        strength: Float,
        method: SpillMethod,
        screenColor: SIMD3<Float>,
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
        encoder.label = "CorridorKey by LateNite Despill"
        encoder.setComputePipelineState(entry.computePipelines.despill)
        encoder.setTexture(foreground, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKDespillParams(
            strength: strength,
            method: method.shaderValue,
            screenColor: screenColor
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
        encoder.label = "CorridorKey by LateNite Levels + Gamma"
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
        encoder.label = "CorridorKey by LateNite Source Passthrough"
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

    // MARK: - Fused matte refine (levels + gamma + refiner blend)

    /// Runs the fused `corridorKeyMatteRefineKernel` — one dispatch
    /// instead of the old `applyAlphaLevelsGamma` → `applyRefinerStrength`
    /// pair. Returns the output texture written (and, when refiner
    /// blend is active, the pooled coarse-matte texture so the caller
    /// can schedule its return).
    ///
    /// Always writes `destination`; does not touch the pool. The coarse
    /// stand-in is generated here via `MatteRefiner.applyGaussianBlur`
    /// when `refinerStrength` ≠ 1.0; otherwise the kernel skips the
    /// read-from-coarse path inside the shader and we pass the matte
    /// itself as a harmless dummy binding.
    static func applyFusedMatteRefine(
        matte: any MTLTexture,
        destination: any MTLTexture,
        blackPoint: Float,
        whitePoint: Float,
        gamma: Float,
        refinerStrength: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (coarsePooled: PooledTexture?, intermediatePooled: PooledTexture?) {
        let width = matte.width
        let height = matte.height

        // Build a coarse stand-in only when the refiner blend will
        // actually use it. Otherwise pass `matte` itself as a dummy —
        // the shader's `abs(strength - 1.0) > 1e-3` guard skips the
        // read so the binding isn't actually consumed.
        var coarseTexture: any MTLTexture = matte
        var coarsePooled: PooledTexture?
        var intermediatePooled: PooledTexture?
        if abs(refinerStrength - 1.0) > 1e-3 {
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
            try MatteRefiner.applyGaussianBlur(
                source: matte,
                intermediate: intermediate.texture,
                destination: coarse.texture,
                radiusPixels: 3.0,
                entry: entry,
                commandBuffer: commandBuffer
            )
            coarseTexture = coarse.texture
            coarsePooled = coarse
            intermediatePooled = intermediate
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            coarsePooled?.returnManually()
            intermediatePooled?.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Matte Refine (Fused)"
        encoder.setComputePipelineState(entry.computePipelines.matteRefineFused)
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(coarseTexture, index: Int(CKTextureIndexCoarse.rawValue))
        encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKMatteRefineParams(
            blackPoint: blackPoint,
            whitePoint: whitePoint,
            gamma: gamma,
            refinerStrength: refinerStrength
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKMatteRefineParams>.size,
            index: 0
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.matteRefineFused,
            width: width,
            height: height
        )
        encoder.endEncoding()
        return (coarsePooled, intermediatePooled)
    }

    // MARK: - Fused foreground post-process

    /// Config bundle for the fused foreground pass. Every field maps
    /// directly onto `CKForegroundPostProcessParams`.
    struct FusedForegroundConfig {
        let sourcePassthrough: Bool
        let lightWrapEnabled: Bool
        let lightWrapStrength: Float
        let lightWrapEdgeBias: Float
        let edgeDecontaminateEnabled: Bool
        let edgeDecontaminateStrength: Float
        let screenColor: SIMD3<Float>
        let inverseScreenMatrix: simd_float3x3
        let applyInverseRotation: Bool
    }

    /// Runs the fused `corridorKeyForegroundPostProcessKernel`. Replaces
    /// the separate `sourcePassthrough`, `applyLightWrap`,
    /// `applyEdgeDecontamination`, and inverse-screen `applyScreenMatrix`
    /// passes with a single compute dispatch. The caller supplies the
    /// pre-blurred source (used only when `lightWrapEnabled` is true);
    /// when light-wrap is off the shader never reads from that slot so
    /// it's safe to bind `sourceRGB` itself as a placeholder.
    static func applyFusedForegroundPostProcess(
        foreground: any MTLTexture,
        sourceRGB: any MTLTexture,
        matte: any MTLTexture,
        blurredSource: any MTLTexture,
        config: FusedForegroundConfig,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture {
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
        encoder.label = "CorridorKey by LateNite Foreground Post-Process (Fused)"
        encoder.setComputePipelineState(entry.computePipelines.foregroundPostProcess)
        encoder.setTexture(foreground, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setTexture(sourceRGB, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(blurredSource, index: Int(CKTextureIndexHint.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))

        var params = CKForegroundPostProcessParams(
            inverseScreenMatrix: config.inverseScreenMatrix,
            screenColor: config.screenColor,
            lightWrapStrength: config.lightWrapStrength,
            lightWrapEdgeBias: config.lightWrapEdgeBias,
            edgeDecontaminateStrength: config.edgeDecontaminateStrength,
            sourcePassthroughEnabled: config.sourcePassthrough ? 1 : 0,
            lightWrapEnabled: config.lightWrapEnabled ? 1 : 0,
            edgeDecontaminateEnabled: config.edgeDecontaminateEnabled ? 1 : 0,
            applyInverseRotation: config.applyInverseRotation ? 1 : 0
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKForegroundPostProcessParams>.size,
            index: 0
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.foregroundPostProcess,
            width: width,
            height: height
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
        encoder.label = "CorridorKey by LateNite Refiner Blend"
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

    // MARK: - Phase 1: Temporal blend (matte flicker reduction)

    /// Blends `currentMatte` toward `previousMatte` on pixels where the
    /// `currentSource`/`previousSource` RGB has barely changed. All four
    /// textures must be the same size and pixel format. Returns a new
    /// pooled texture holding the blended matte.
    ///
    /// `strength` is the exponential-moving-average weight assigned to the
    /// previous frame when the pixel is deemed stationary (0 disables the
    /// blend entirely; 1 replaces the current alpha with the previous
    /// alpha). `motionThreshold` is the max-channel absolute RGB delta at
    /// which the gate drops to half strength; values at 2× threshold pass
    /// the current alpha through unchanged. See the matching kernel in
    /// `CorridorKeyShaders.metal` for the exact math.
    ///
    /// Returns `nil` when `strength <= 0` — the caller should use
    /// `currentMatte` directly to avoid an unnecessary encoder round-trip.
    static func applyTemporalBlend(
        currentMatte: any MTLTexture,
        previousMatte: any MTLTexture,
        currentSource: any MTLTexture,
        previousSource: any MTLTexture,
        strength: Float,
        motionThreshold: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws -> PooledTexture? {
        if strength <= 0 { return nil }
        precondition(
            currentMatte.width == previousMatte.width
                && currentMatte.height == previousMatte.height,
            "Matte textures must share dimensions for temporal blend"
        )
        precondition(
            currentSource.width == currentMatte.width
                && currentSource.height == currentMatte.height
                && previousSource.width == currentMatte.width
                && previousSource.height == currentMatte.height,
            "Source and matte must share dimensions for temporal blend"
        )

        let width = currentMatte.width
        let height = currentMatte.height

        guard let output = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: currentMatte.pixelFormat
        ) else {
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            output.returnManually()
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "CorridorKey by LateNite Temporal Blend"
        encoder.setComputePipelineState(entry.computePipelines.temporalBlend)
        encoder.setTexture(currentMatte, index: Int(CKTextureIndexMatte.rawValue))
        encoder.setTexture(previousMatte, index: Int(CKTextureIndexPreviousMatte.rawValue))
        encoder.setTexture(currentSource, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(previousSource, index: Int(CKTextureIndexPreviousSource.rawValue))
        encoder.setTexture(output.texture, index: Int(CKTextureIndexOutput.rawValue))
        var params = CKTemporalBlendParams(
            strength: strength,
            motionThreshold: max(motionThreshold, 1e-6)
        )
        encoder.setBytes(
            &params,
            length: MemoryLayout<CKTemporalBlendParams>.size,
            index: Int(CKBufferIndexTemporalBlendParams.rawValue)
        )
        dispatch(
            encoder: encoder,
            pipeline: entry.computePipelines.temporalBlend,
            width: width,
            height: height
        )
        encoder.endEncoding()
        return output
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
        //
        // `matteThreshold` deliberately sits at a low 0.1 — components are
        // taken to include every pixel the model thinks *might* be
        // foreground. This keeps soft hair strands and transparent haloes
        // attached to their parent subject instead of being amputated by a
        // hard 0.5 cut, so the filter only deletes specks that are actually
        // isolated from the subject.
        var params = CKCCLabelParams(
            areaThreshold: Int32(areaThreshold),
            labelSpan: Int32(width * height + 1),
            matteThreshold: 0.1,
            blurSigma: 0.0
        )

        // Init pass: binarise matte → unique integer label per pixel.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CorridorKey by LateNite CC Init"
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

        // Propagation phase 1: min-neighbour flood with 1-pixel stride.
        // 32 iterations is plenty to establish a full "chain" within every
        // component — every pixel's label now points (transitively) to its
        // component's global minimum. Each iteration is an 8-tap
        // neighbourhood read, so on 4K this is under ~2 ms even at the
        // high count.
        let maxDim = max(width, height)
        let propagateIterations = min(32, max(8, maxDim / 16))
        var sourceLabel = labelA
        var destinationLabel = labelB
        for _ in 0..<propagateIterations {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "CorridorKey by LateNite CC Propagate"
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

        // Propagation phase 2: classic doubling pointer-jump. Each
        // iteration follows the parent link from each pixel's current
        // label, so chains of length L collapse in ~log₂(L) iterations.
        // `log₂(width * height)` caps the worst case (say 24 iterations
        // on a 4K matte) which is still faster than adding a further few
        // hundred stride-1 passes and — crucially — correct for large
        // subjects where stride-1 alone can't reach the minimum label.
        let pointerJumpIterations = max(8, Int(ceil(log2(Double(width * height + 1)))))
        var widthParam = Int32(width)
        for _ in 0..<pointerJumpIterations {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "CorridorKey by LateNite CC Pointer Jump"
                encoder.setComputePipelineState(entry.computePipelines.ccLabelPointerJump)
                encoder.setTexture(sourceLabel.texture, index: Int(CKTextureIndexMatte.rawValue))
                encoder.setTexture(destinationLabel.texture, index: Int(CKTextureIndexOutput.rawValue))
                encoder.setBytes(&widthParam, length: MemoryLayout<Int32>.size, index: 0)
                dispatch(
                    encoder: encoder,
                    pipeline: entry.computePipelines.ccLabelPointerJump,
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
        //
        // Pulled from the device-cache pool so the despeckle pass no
        // longer allocates ~64 MB per 4K frame. The buffer is shared
        // across frames at the same (width, height), with a fresh
        // GPU-side `fill(buffer:range:value:0)` zeroing it at the
        // start of every pass — kernel semantics unchanged.
        guard let countsBuffer = entry.connectedComponentsCountsBuffer(width: width, height: height) else {
            labelA.returnManually()
            labelB.returnManually()
            throw MetalDeviceCacheError.textureAllocationFailed
        }
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "CorridorKey by LateNite CC Counts Zero"
            blit.fill(buffer: countsBuffer, range: 0..<countsBuffer.length, value: 0)
            blit.endEncoding()
        }

        // Atomic count pass.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CorridorKey by LateNite CC Count"
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
            encoder.label = "CorridorKey by LateNite CC Filter"
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
        encoder.label = "CorridorKey by LateNite Light Wrap"
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
        encoder.label = "CorridorKey by LateNite Edge Decontaminate"
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
