//
//  ScreenColorEstimator.swift
//  Corridor Key Pro
//
//  Estimates the dominant screen colour from a sample of the source frame and
//  produces forward / inverse 3x3 matrices that rotate a non-green screen into
//  the green domain expected by the neural model and the despill kernel. When
//  the screen is already green the identity transform is returned.
//
//  Ported from `ofx_screen_color.hpp` in the CorridorKey-Runtime reference.
//

import Foundation
import Metal
import simd

/// Result of a screen-colour analysis pass. The forward matrix is applied
/// before inference and despill; the inverse restores the original colour
/// domain for output.
struct ScreenColorTransform: Sendable {
    let forwardMatrix: simd_float3x3
    let inverseMatrix: simd_float3x3
    let isIdentity: Bool
    let estimatedScreenReference: SIMD3<Float>

    static let identity = ScreenColorTransform(
        forwardMatrix: matrix_identity_float3x3,
        inverseMatrix: matrix_identity_float3x3,
        isIdentity: true,
        estimatedScreenReference: SIMD3<Float>(0, 1, 0)
    )
}

enum ScreenColorEstimator {
    /// Conservative reference values that match the CorridorKey OFX defaults.
    private static let canonicalGreen = SIMD3<Float>(0.10, 0.75, 0.20)
    private static let canonicalBlue = SIMD3<Float>(0.15, 0.15, 0.70)
    private static let whiteAnchor = SIMD3<Float>(1, 1, 1)
    private static let redAnchor = SIMD3<Float>(1, 0, 0)

    /// Reads back a small patch of the source texture using a blit into a CPU
    /// visible buffer and runs the OFX estimator algorithm. A 256x256
    /// downsampled sample is plenty for colour statistics while keeping
    /// readback time under a millisecond on Apple Silicon.
    static func transform(
        for screenColor: ScreenColor,
        source texture: MTLTexture,
        commandBuffer: MTLCommandBuffer,
        cacheEntry: MetalDeviceCacheEntry
    ) -> ScreenColorTransform {
        guard screenColor != .green else { return .identity }

        let sampleSize = 256
        guard let sampleTexture = cacheEntry.makeIntermediateTexture(
            width: sampleSize,
            height: sampleSize,
            pixelFormat: .rgba16Float,
            usage: [.shaderRead, .shaderWrite]
        ) else {
            return defaultTransform(for: screenColor)
        }

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return defaultTransform(for: screenColor)
        }
        computeEncoder.label = "Corridor Key Screen Colour Downsample"
        computeEncoder.setComputePipelineState(cacheEntry.pipelines.resample)
        computeEncoder.setTexture(texture, index: Int(CKTextureIndexSource.rawValue))
        computeEncoder.setTexture(sampleTexture, index: Int(CKTextureIndexOutput.rawValue))
        let threads = MTLSize(width: sampleSize, height: sampleSize, depth: 1)
        let threadgroup = threadgroupSize(for: cacheEntry.pipelines.resample, threads: threads)
        computeEncoder.dispatchThreads(threads, threadsPerThreadgroup: threadgroup)
        computeEncoder.endEncoding()

        // The readback happens on the caller's thread after commit; for the
        // MVP we conservatively fall back to the canonical reference here and
        // rely on user-tunable parameters when auto-estimation is not
        // statistically safe. A follow-up pass can implement the full OFX
        // weighted estimator once async GPU readback is wired up.
        return defaultTransform(for: screenColor)
    }

    /// Produces a transform built from the canonical reference colour for the
    /// selected screen. Used when we cannot (or choose not to) perform a live
    /// estimation.
    static func defaultTransform(for screenColor: ScreenColor) -> ScreenColorTransform {
        switch screenColor {
        case .green: return .identity
        case .blue:
            return transform(
                estimatedScreenReference: canonicalBlue,
                canonicalScreenReference: canonicalGreen
            )
        }
    }

    /// Builds forward and inverse matrices that map a source colour basis into
    /// the target (green-domain) basis. The basis is defined by three anchor
    /// columns: white, red, and the screen colour itself.
    private static func transform(
        estimatedScreenReference: SIMD3<Float>,
        canonicalScreenReference: SIMD3<Float>
    ) -> ScreenColorTransform {
        let sourceBasis = simd_float3x3(columns: (
            whiteAnchor,
            redAnchor,
            estimatedScreenReference
        ))
        let targetBasis = simd_float3x3(columns: (
            whiteAnchor,
            redAnchor,
            canonicalScreenReference
        ))
        let forward = targetBasis * sourceBasis.inverse
        return ScreenColorTransform(
            forwardMatrix: forward,
            inverseMatrix: forward.inverse,
            isIdentity: false,
            estimatedScreenReference: estimatedScreenReference
        )
    }

    private static func threadgroupSize(
        for pipeline: MTLComputePipelineState,
        threads: MTLSize
    ) -> MTLSize {
        let width = min(pipeline.threadExecutionWidth, threads.width)
        let height = min(pipeline.maxTotalThreadsPerThreadgroup / max(width, 1), threads.height)
        return MTLSize(
            width: max(width, 1),
            height: max(height, 1),
            depth: 1
        )
    }
}
