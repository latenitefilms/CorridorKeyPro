//
//  TemporalBlender.swift
//  CorridorKey by LateNite
//
//  CPU-side mirror of `corridorKeyTemporalBlendKernel`. Applied during the
//  analysis pass so the cached matte already reflects the blend — playback
//  renders read the stabilised matte from the custom-parameter cache with
//  zero extra work on the hot path.
//
//  Inputs arrive as flat float buffers from `RenderPipeline.extractAlphaMatteForAnalysis`
//  after a one-time GPU → CPU readback (the same round-trip we already make
//  to persist the alpha); running the blend on the CPU avoids an additional
//  command buffer round-trip and keeps the plug-in's lifetime map of
//  persistent GPU textures narrow (the analyser only holds flat Float arrays
//  between frames).
//

import Foundation
import simd

public enum TemporalBlender {

    /// Parameters for a single blend call.
    public struct Configuration: Sendable, Equatable {
        /// Weight assigned to the previous frame's alpha when the pixel is
        /// deemed stationary. `0` bypasses the blend entirely; `1` replaces
        /// the current alpha with the previous alpha on stationary pixels.
        public let strength: Float
        /// Max-channel absolute RGB delta at which the gate reaches zero.
        /// Pixels above `2 × motionThreshold` pass the current alpha through
        /// unchanged; linear falloff in between.
        public let motionThreshold: Float

        public init(strength: Float, motionThreshold: Float) {
            self.strength = strength
            self.motionThreshold = motionThreshold
        }

        /// Ships a `nil` configuration-equivalent when the user disables
        /// temporal stability — saves a branch at every call site.
        public var isNoOp: Bool { strength <= 0 }
    }

    /// Single-pixel blend. Extracted so tests can assert on the math without
    /// constructing full-frame arrays.
    public static func blendPixel(
        currentAlpha: Float,
        previousAlpha: Float,
        currentRGB: SIMD3<Float>,
        previousRGB: SIMD3<Float>,
        configuration: Configuration
    ) -> Float {
        if configuration.isNoOp { return currentAlpha }
        let delta = simd_abs(currentRGB - previousRGB)
        let motion = max(delta.x, max(delta.y, delta.z))
        let threshold = max(configuration.motionThreshold, 1e-6)
        let motionGate = max(Float(0), min(Float(1), 1 - motion / (2 * threshold)))
        let effectiveStrength = max(Float(0), min(Float(1), configuration.strength)) * motionGate
        let blended = currentAlpha + (previousAlpha - currentAlpha) * effectiveStrength
        return min(1, max(0, blended))
    }

    /// Applies the EMA blend in place on `currentAlpha`. `previousAlpha`,
    /// `currentSource`, and `previousSource` must all describe the same
    /// `width × height` grid. `currentSource` and `previousSource` are
    /// expected in interleaved RGBA-32F layout (4 floats per pixel); the
    /// alpha channel of the source is ignored — only the RGB delta drives
    /// the motion gate.
    public static func applyInPlace(
        currentAlpha: inout [Float],
        previousAlpha: [Float],
        currentSource: [Float],
        previousSource: [Float],
        width: Int,
        height: Int,
        configuration: Configuration
    ) {
        if configuration.isNoOp { return }
        let pixelCount = width * height
        precondition(currentAlpha.count == pixelCount, "currentAlpha size mismatch")
        precondition(previousAlpha.count == pixelCount, "previousAlpha size mismatch")
        precondition(currentSource.count == pixelCount * 4, "currentSource size mismatch")
        precondition(previousSource.count == pixelCount * 4, "previousSource size mismatch")

        let strength: Float = max(0, min(1, configuration.strength))
        let threshold: Float = max(configuration.motionThreshold, 1e-6)
        let invDenominator: Float = 1 / (2 * threshold)

        currentAlpha.withUnsafeMutableBufferPointer { alphaPointer in
            previousAlpha.withUnsafeBufferPointer { previousAlphaPointer in
                currentSource.withUnsafeBufferPointer { currentSourcePointer in
                    previousSource.withUnsafeBufferPointer { previousSourcePointer in
                        guard
                            let alphaBase = alphaPointer.baseAddress,
                            let previousAlphaBase = previousAlphaPointer.baseAddress,
                            let currentSourceBase = currentSourcePointer.baseAddress,
                            let previousSourceBase = previousSourcePointer.baseAddress
                        else { return }
                        for pixelIndex in 0..<pixelCount {
                            let sourceIndex = pixelIndex * 4
                            let deltaR = abs(currentSourceBase[sourceIndex] - previousSourceBase[sourceIndex])
                            let deltaG = abs(currentSourceBase[sourceIndex + 1] - previousSourceBase[sourceIndex + 1])
                            let deltaB = abs(currentSourceBase[sourceIndex + 2] - previousSourceBase[sourceIndex + 2])
                            let motion = max(deltaR, max(deltaG, deltaB))
                            let gate = max(Float(0), min(Float(1), 1 - motion * invDenominator))
                            let effectiveStrength = strength * gate
                            let current = alphaBase[pixelIndex]
                            let previous = previousAlphaBase[pixelIndex]
                            let blended = current + (previous - current) * effectiveStrength
                            alphaBase[pixelIndex] = min(1, max(0, blended))
                        }
                    }
                }
            }
        }
    }
}
