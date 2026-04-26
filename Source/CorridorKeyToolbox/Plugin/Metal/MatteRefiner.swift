//
//  MatteRefiner.swift
//  CorridorKey by LateNite
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
/// threadgroup-cached separable kernel. Re-benchmarked after the kernel
/// migrated from `texture.sample` to threadgroup-shared loads + `read`:
/// the custom path now wins from radius 1 up through the threadgroup
/// cache cap. Beyond that MPS's running-window O(1) kernel still wins,
/// so the breakeven is the cache cap.
private let mpsRadiusBreakeven = morphologyTGCacheRadius + 1

/// Minimum Gaussian sigma at which `MPSImageGaussianBlur` beats our
/// threadgroup-cached separable compute kernel. Re-benchmarked after
/// the kernel switched to `texture.read` + threadgroup memory; for
/// radii within the cache cap our path matches MPS, beyond that MPS
/// wins. `applyGaussianBlur` clamps the kernel radius to ≤ cache cap
/// before deciding, so this constant only matters for very high
/// sigmas (driven by user-facing softness sliders).
private let mpsSigmaBreakeven: Float = Float(morphologyTGCacheRadius)

/// Mirrors `kMorphMaxRadius` in `CorridorKeyShaders.metal` — the
/// largest radius the threadgroup cache can hold. Beyond this the
/// dispatcher must route to MPS.
let morphologyTGCacheRadius = 32

/// Mirrors `kMorphTGWidth` in `CorridorKeyShaders.metal`. Threadgroup
/// width along the active axis (x for horizontal, y for vertical).
let morphologyTGWidth = 64

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
            encoder.label = "CorridorKey by LateNite Morphology H"
            encoder.setComputePipelineState(entry.computePipelines.morphologyHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&absoluteRadius, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            dispatchSeparableStrip(
                encoder: encoder,
                width: intermediate.width,
                height: intermediate.height,
                axis: .horizontal
            )
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CorridorKey by LateNite Morphology V"
            encoder.setComputePipelineState(entry.computePipelines.morphologyVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBytes(&absoluteRadius, length: MemoryLayout<Int32>.size, index: 0)
            encoder.setBytes(&erodeFlag, length: MemoryLayout<Int32>.size, index: 1)
            dispatchSeparableStrip(
                encoder: encoder,
                width: destination.width,
                height: destination.height,
                axis: .vertical
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
            encoder.label = "CorridorKey by LateNite Blur H"
            encoder.setComputePipelineState(entry.computePipelines.gaussianHorizontal)
            encoder.setTexture(source, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(intermediate, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weights.buffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            dispatchSeparableStrip(
                encoder: encoder,
                width: intermediate.width,
                height: intermediate.height,
                axis: .horizontal
            )
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CorridorKey by LateNite Blur V"
            encoder.setComputePipelineState(entry.computePipelines.gaussianVertical)
            encoder.setTexture(intermediate, index: Int(CKTextureIndexMatte.rawValue))
            encoder.setTexture(destination, index: Int(CKTextureIndexOutput.rawValue))
            encoder.setBuffer(weights.buffer, offset: 0, index: Int(CKBufferIndexBlurWeights.rawValue))
            encoder.setBytes(&radiusValue, length: MemoryLayout<Int32>.size, index: 0)
            dispatchSeparableStrip(
                encoder: encoder,
                width: destination.width,
                height: destination.height,
                axis: .vertical
            )
            encoder.endEncoding()
        }
    }

    /// Axis selector for `dispatchSeparableStrip`. `.horizontal` arranges
    /// the threadgroup along the x-axis; `.vertical` along y. The shader
    /// uses `tg_thread.x` or `tg_thread.y` accordingly.
    private enum SeparableAxis {
        case horizontal
        case vertical
    }

    /// Dispatch helper for the threadgroup-cached separable kernels
    /// (`corridorKeyMorphology*` and `corridorKeyGaussian*`). Uses
    /// `dispatchThreadgroups(_:threadsPerThreadgroup:)` with an explicit
    /// `morphologyTGWidth`-thread layout along the active axis so the
    /// shader's threadgroup-shared cache layout matches the dispatch.
    /// Generic `dispatchThreads(_:threadsPerThreadgroup:)` lets Metal
    /// pick threadgroup widths that don't match the shader's cooperative
    /// load assumptions, which produces undefined results.
    private static func dispatchSeparableStrip(
        encoder: any MTLComputeCommandEncoder,
        width: Int,
        height: Int,
        axis: SeparableAxis
    ) {
        let safeWidth = max(width, 1)
        let safeHeight = max(height, 1)
        let stripCount = (axis == .horizontal ? safeWidth : safeHeight)
        let stripGroups = (stripCount + morphologyTGWidth - 1) / morphologyTGWidth
        let threadsPerThreadgroup: MTLSize
        let threadgroupsPerGrid: MTLSize
        switch axis {
        case .horizontal:
            threadsPerThreadgroup = MTLSize(width: morphologyTGWidth, height: 1, depth: 1)
            threadgroupsPerGrid = MTLSize(width: stripGroups, height: safeHeight, depth: 1)
        case .vertical:
            threadsPerThreadgroup = MTLSize(width: 1, height: morphologyTGWidth, depth: 1)
            threadgroupsPerGrid = MTLSize(width: safeWidth, height: stripGroups, depth: 1)
        }
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }
}
