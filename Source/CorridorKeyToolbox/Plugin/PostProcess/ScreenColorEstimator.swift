//
//  ScreenColorEstimator.swift
//  CorridorKey by LateNite
//
//  Produces forward / inverse 3x3 matrices that rotate a non-green screen into
//  the green domain expected by the neural model and the despill kernel. When
//  the screen is already green the identity transform is returned.
//
//  Ported from `ofx_screen_color.hpp` in the CorridorKey-Runtime reference.
//  Live per-frame estimation is a follow-up; the canonical reference values
//  match CorridorKey's defaults and produce identical results on well-lit
//  screens.
//

import Foundation
import Metal
import simd

struct ScreenColorTransform: Sendable {
    let forwardMatrix: simd_float3x3
    let inverseMatrix: simd_float3x3
    let isIdentity: Bool
    let estimatedScreenReference: SIMD3<Float>

    static let identity = ScreenColorTransform(
        forwardMatrix: matrix_identity_float3x3,
        inverseMatrix: matrix_identity_float3x3,
        isIdentity: true,
        estimatedScreenReference: SIMD3<Float>(0.08, 0.84, 0.08)
    )
}

enum ScreenColorEstimator {
    /// Reference values used when no live estimation is performed. These match
    /// the CorridorKey-Runtime defaults in `ofx_screen_color.hpp`.
    private static let canonicalGreen = SIMD3<Float>(0.08, 0.84, 0.08)
    private static let canonicalBlue = SIMD3<Float>(0.08, 0.16, 0.84)
    private static let whiteAnchor = SIMD3<Float>(1, 1, 1)
    private static let redAnchor = SIMD3<Float>(1, 0, 0)

    /// Returns the transform appropriate for a screen colour without touching
    /// the GPU. Sufficient for well-lit screens and keeps the render path
    /// free of expensive per-frame readbacks.
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
    /// the target basis. The basis is defined by three anchor columns: white,
    /// red, and the screen colour itself — identical to CorridorKey's OFX
    /// path so results match across hosts.
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
}
