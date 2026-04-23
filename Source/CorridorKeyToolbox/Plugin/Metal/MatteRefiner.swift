//
//  MatteRefiner.swift
//  Corridor Key Toolbox
//
//  Radius-aware dispatcher that runs morphology (erode / dilate) and Gaussian
//  blur through `MetalPerformanceShaders` when the kernel is big enough to
//  benefit from it, and falls back to our own separable compute kernels for
//  tiny radii where MPS setup overhead outweighs its speed-up.
//
//  MPS uses threadgroup-shared memory plus a running-window algorithm for
//  morphology, so at radius ≥ 3 it beats the naive separable loop in
//  `corridorKeyMorphology*Kernel`. Gaussian blur via `MPSImageGaussianBlur`
//  is similarly faster at σ > 1.5 and produces higher-quality results
//  thanks to its optimised tap pattern.
//
//  Lanczos upscale (`MPSImageLanczosScale`) replaces the previous bilinear
//  compute kernel when the user picks the Lanczos Quality Mode. For tiny
//  resizes (within 20% of 1:1) MPS Lanczos is slower than bilinear and
//  visually equivalent — the caller gets to request the specific method.
//

import Foundation
import Metal
import MetalPerformanceShaders

/// Minimum kernel radius (in texels) at which MPS dilate/erode beats the
/// custom separable kernel. Chosen by benchmarking the shipping shaders on
/// M2 Pro and M3 Max — below radius 3 the MPS setup cost dominates.
private let mpsRadiusBreakeven = 3

/// Minimum Gaussian sigma at which `MPSImageGaussianBlur` beats our
/// separable compute kernel. `MPSImageGaussianBlur(sigma: < 1)` tends to fall
/// through to a bilinear-ish fast path internally anyway; we leave small
/// blurs in our weighted-tap kernel so the weights cache stays warm.
private let mpsSigmaBreakeven: Float = 1.5

enum MatteRefiner {

    // MARK: - Morphology

    /// Erode or dilate the matte texture by `radius` pixels. Writes into
    /// `destination`. `intermediate` is used as ping-pong scratch for the
    /// two-axis separable fallback; it is unused by the MPS path but still
    /// required by the caller's bookkeeping so the function signature stays
    /// stable across branches.
    ///
    /// - Parameters:
    ///   - radius: positive value dilates, negative erodes, zero is a no-op.
    static func applyMorphology(
        source: any MTLTexture,
        intermediate: any MTLTexture,
        destination: any MTLTexture,
        radius: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let absRadius = abs(radius)
        guard absRadius > 0 else { return }
        let isErode = radius < 0

        if absRadius >= mpsRadiusBreakeven {
            let kernelSide = 2 * absRadius + 1
            if isErode {
                if let erode = entry.mpsErode(kernelSide: kernelSide) {
                    erode.encode(
                        commandBuffer: commandBuffer,
                        sourceTexture: source,
                        destinationTexture: destination
                    )
                    return
                }
            } else {
                if let dilate = entry.mpsDilate(kernelSide: kernelSide) {
                    dilate.encode(
                        commandBuffer: commandBuffer,
                        sourceTexture: source,
                        destinationTexture: destination
                    )
                    return
                }
            }
        }

        try runCustomMorphology(
            source: source,
            intermediate: intermediate,
            destination: destination,
            radius: radius,
            entry: entry,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - Gaussian blur

    /// Applies a separable Gaussian blur with the supplied radius (in pixels)
    /// and matching sigma. `intermediate` is used by the compute fallback for
    /// the horizontal→vertical ping-pong; the MPS path doesn't need it but
    /// the argument is still required for API consistency.
    static func applyGaussianBlur(
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
        if sigma >= mpsSigmaBreakeven, let blur = entry.mpsGaussianBlur(sigma: sigma) {
            blur.encode(
                commandBuffer: commandBuffer,
                sourceTexture: source,
                destinationTexture: destination
            )
            return
        }

        try runCustomGaussianBlur(
            source: source,
            intermediate: intermediate,
            destination: destination,
            kernelRadius: kernelRadius,
            sigma: sigma,
            entry: entry,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - Lanczos resample (used when Quality = Lanczos)

    /// Resamples `source` to fit the dimensions of `destination` using
    /// `MPSImageLanczosScale`. The caller ensures the textures are of the
    /// correct size before calling. Pixel formats must match what MPS
    /// supports (both `.rgba16Float`, both `.rgba32Float`, both `.r16Float`,
    /// both `.r32Float`, etc.). Source/destination can share no memory.
    static func applyLanczosResample(
        source: any MTLTexture,
        destination: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) {
        let scaler = entry.mpsLanczosScale()
        scaler.edgeMode = .clamp
        scaler.encode(
            commandBuffer: commandBuffer,
            sourceTexture: source,
            destinationTexture: destination
        )
    }

    // MARK: - Private compute fallbacks

    private static func runCustomMorphology(
        source: any MTLTexture,
        intermediate: any MTLTexture,
        destination: any MTLTexture,
        radius: Int,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        var absoluteRadius = Int32(abs(radius))
        var erodeFlag: Int32 = radius < 0 ? 1 : 0

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Morphology H"
            encoder.setComputePipelineState(entry.computePipelines.morphologyHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&absoluteRadius, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            dispatchThreads(
                encoder: encoder,
                pipeline: entry.computePipelines.morphologyHorizontal,
                width: intermediate.width,
                height: intermediate.height
            )
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Morphology V"
            encoder.setComputePipelineState(entry.computePipelines.morphologyVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&absoluteRadius, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            dispatchThreads(
                encoder: encoder,
                pipeline: entry.computePipelines.morphologyVertical,
                width: destination.width,
                height: destination.height
            )
            encoder.endEncoding()
        }
    }

    private static func runCustomGaussianBlur(
        source: any MTLTexture,
        intermediate: any MTLTexture,
        destination: any MTLTexture,
        kernelRadius: Int,
        sigma: Float,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let weights = entry.gaussianWeightsBuffer(radius: kernelRadius, sigma: sigma) else { return }
        var radiusValue = Int32(kernelRadius)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Blur H"
            encoder.setComputePipelineState(entry.computePipelines.gaussianHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weights.buffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            dispatchThreads(
                encoder: encoder,
                pipeline: entry.computePipelines.gaussianHorizontal,
                width: intermediate.width,
                height: intermediate.height
            )
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Corridor Key Toolbox Blur V"
            encoder.setComputePipelineState(entry.computePipelines.gaussianVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weights.buffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            dispatchThreads(
                encoder: encoder,
                pipeline: entry.computePipelines.gaussianVertical,
                width: destination.width,
                height: destination.height
            )
            encoder.endEncoding()
        }
    }

    private static func dispatchThreads(
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
