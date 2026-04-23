//
//  ScreenColorEstimatorTests.swift
//  CorridorKeyToolboxLogicTests
//
//  The render pipeline applies the forward matrix before inference and the
//  inverse on the way out. If forward × inverse drifts from identity the
//  output picks up a colour cast that isn't obvious until a real green-screen
//  clip is loaded in Final Cut Pro — so pin the round-trip numerically here.
//

import Testing
import simd
@testable import CorridorKeyToolboxLogic

@Suite("ScreenColorEstimator")
struct ScreenColorEstimatorTests {

    @Test("Green returns the identity transform")
    func greenIsIdentity() {
        let transform = ScreenColorEstimator.defaultTransform(for: .green)
        #expect(transform.isIdentity == true)
        expectMatricesEqual(transform.forwardMatrix, matrix_identity_float3x3)
        expectMatricesEqual(transform.inverseMatrix, matrix_identity_float3x3)
    }

    @Test("Blue inverts cleanly")
    func blueInverseRoundTrips() {
        let transform = ScreenColorEstimator.defaultTransform(for: .blue)
        #expect(transform.isIdentity == false)

        let composed = transform.inverseMatrix * transform.forwardMatrix
        expectMatricesEqual(composed, matrix_identity_float3x3, tolerance: 1e-4)
    }

    @Test("Blue maps a canonical blue pixel into the green domain")
    func blueForwardMapsCanonicalBlue() {
        let transform = ScreenColorEstimator.defaultTransform(for: .blue)
        let canonicalBlue = SIMD3<Float>(0.08, 0.16, 0.84)
        let canonicalGreen = SIMD3<Float>(0.08, 0.84, 0.08)
        let mapped = transform.forwardMatrix * canonicalBlue
        expectVectorsEqual(mapped, canonicalGreen, tolerance: 1e-4)
    }

    // MARK: - Helpers

    private func expectMatricesEqual(
        _ lhs: simd_float3x3,
        _ rhs: simd_float3x3,
        tolerance: Float = 1e-5
    ) {
        for column in 0..<3 {
            for row in 0..<3 {
                let delta = abs(lhs[column][row] - rhs[column][row])
                #expect(delta < tolerance, "Mismatch at [\(column)][\(row)]: \(lhs[column][row]) vs \(rhs[column][row])")
            }
        }
    }

    private func expectVectorsEqual(
        _ lhs: SIMD3<Float>,
        _ rhs: SIMD3<Float>,
        tolerance: Float = 1e-5
    ) {
        for index in 0..<3 {
            let delta = abs(lhs[index] - rhs[index])
            #expect(delta < tolerance, "Mismatch at [\(index)]: \(lhs[index]) vs \(rhs[index])")
        }
    }
}
