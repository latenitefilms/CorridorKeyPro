//
//  TemporalBlenderTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Covers the CPU-side `TemporalBlender` that runs during the analysis
//  pass. The math must agree with `corridorKeyTemporalBlendKernel`
//  (golden-tested in `TemporalBlendTests` of the Metal suite) so swapping
//  in the GPU implementation later — or reversing to CPU on a slow GPU —
//  stays drop-in compatible.
//

import Foundation
import Testing
import simd
@testable import CorridorKeyToolboxLogic

@Suite struct TemporalBlenderTests {

    // MARK: - Pixel-level math

    @Test func stationaryPixelAtFullStrengthReplacesAlpha() {
        let configuration = TemporalBlender.Configuration(strength: 1.0, motionThreshold: 0.05)
        let rgb = SIMD3<Float>(0.5, 0.5, 0.5)
        let result = TemporalBlender.blendPixel(
            currentAlpha: 0.9,
            previousAlpha: 0.1,
            currentRGB: rgb,
            previousRGB: rgb,
            configuration: configuration
        )
        #expect(abs(result - 0.1) < 1e-6)
    }

    @Test func stationaryPixelAtHalfStrengthAverages() {
        let configuration = TemporalBlender.Configuration(strength: 0.5, motionThreshold: 0.05)
        let rgb = SIMD3<Float>(0.2, 0.3, 0.4)
        let result = TemporalBlender.blendPixel(
            currentAlpha: 0.8,
            previousAlpha: 0.4,
            currentRGB: rgb,
            previousRGB: rgb,
            configuration: configuration
        )
        // current + (prev - current) * 0.5 = 0.8 + -0.4 * 0.5 = 0.6
        #expect(abs(result - 0.6) < 1e-6)
    }

    @Test func largeMotionBypassesBlend() {
        let configuration = TemporalBlender.Configuration(strength: 1.0, motionThreshold: 0.05)
        let result = TemporalBlender.blendPixel(
            currentAlpha: 0.75,
            previousAlpha: 0.1,
            currentRGB: SIMD3<Float>(0.9, 0.5, 0.5),
            previousRGB: SIMD3<Float>(0.2, 0.5, 0.5),
            configuration: configuration
        )
        // Motion = 0.7, way above 2 × 0.05 → gate 0 → alpha unchanged.
        #expect(abs(result - 0.75) < 1e-6)
    }

    @Test func motionAtThresholdHalvesStrength() {
        let threshold: Float = 0.05
        let configuration = TemporalBlender.Configuration(strength: 1.0, motionThreshold: threshold)
        let result = TemporalBlender.blendPixel(
            currentAlpha: 1.0,
            previousAlpha: 0.0,
            currentRGB: SIMD3<Float>(0.5, 0.5, 0.5),
            previousRGB: SIMD3<Float>(0.5 + threshold, 0.5, 0.5),
            configuration: configuration
        )
        // Gate = 1 − 0.05 / (2·0.05) = 0.5 → effectiveStrength = 0.5
        // → result = 1.0 + (0 − 1) · 0.5 = 0.5
        #expect(abs(result - 0.5) < 1e-6)
    }

    @Test func zeroStrengthShortCircuits() {
        let configuration = TemporalBlender.Configuration(strength: 0, motionThreshold: 0.05)
        let rgb = SIMD3<Float>(0.5, 0.5, 0.5)
        let result = TemporalBlender.blendPixel(
            currentAlpha: 0.37,
            previousAlpha: 0.91,
            currentRGB: rgb,
            previousRGB: rgb,
            configuration: configuration
        )
        #expect(result == 0.37)
    }

    // MARK: - Full-frame in-place blend

    @Test func applyInPlaceBlendsEachPixelIndependently() {
        // Build a 4×1 frame with a mix of stationary and moving pixels so a
        // single call exercises the motion gate, the stationary blend and
        // the fast-path bail-out in one sweep.
        let configuration = TemporalBlender.Configuration(strength: 0.5, motionThreshold: 0.05)
        var currentAlpha: [Float] = [0.8, 0.3, 1.0, 0.5]
        let previousAlpha: [Float] = [0.4, 0.1, 0.0, 0.5]
        let currentSource: [Float] = [
            0.5, 0.5, 0.5, 1,  // stationary
            0.5, 0.5, 0.5, 1,  // stationary
            0.9, 0.5, 0.5, 1,  // moving (Δ 0.4)
            0.5, 0.5, 0.5, 1   // stationary
        ]
        let previousSource: [Float] = [
            0.5, 0.5, 0.5, 1,
            0.5, 0.5, 0.5, 1,
            0.2, 0.5, 0.5, 1,
            0.5, 0.5, 0.5, 1
        ]
        TemporalBlender.applyInPlace(
            currentAlpha: &currentAlpha,
            previousAlpha: previousAlpha,
            currentSource: currentSource,
            previousSource: previousSource,
            width: 4,
            height: 1,
            configuration: configuration
        )

        // Pixels 0, 1, 3 are stationary → strength 0.5 → current + (prev − current)·0.5
        #expect(abs(currentAlpha[0] - 0.6) < 1e-6)
        #expect(abs(currentAlpha[1] - 0.2) < 1e-6)
        // Pixel 2 moves by 0.4 → gate saturates to 0 → alpha unchanged.
        #expect(abs(currentAlpha[2] - 1.0) < 1e-6)
        // Pixel 3: current == previous, blend is a no-op.
        #expect(abs(currentAlpha[3] - 0.5) < 1e-6)
    }

    @Test func noOpConfigurationLeavesArrayUnchanged() {
        let configuration = TemporalBlender.Configuration(strength: 0, motionThreshold: 0.05)
        var currentAlpha: [Float] = [0.1, 0.2, 0.3, 0.4]
        let before = currentAlpha
        let sourceFrame: [Float] = Array(repeating: 0.5, count: 4 * 4)
        TemporalBlender.applyInPlace(
            currentAlpha: &currentAlpha,
            previousAlpha: [0.9, 0.9, 0.9, 0.9],
            currentSource: sourceFrame,
            previousSource: sourceFrame,
            width: 4,
            height: 1,
            configuration: configuration
        )
        #expect(currentAlpha == before)
    }
}
