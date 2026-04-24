//
//  TemporalAccumulatorTests.swift
//  BenchmarkRunnerTests
//
//  Exercises the streaming statistics layer with synthetic matte
//  sequences. The disk-backed `TemporalMetricsCalculator.compute`
//  path just drives this accumulator, so covering it thoroughly is
//  where the temporal-coherence math gets its real validation.
//

import Testing
@testable import BenchmarkRunner

@Suite struct TemporalAccumulatorTests {

    @Test func constantFrameSequenceReportsZeroStdDev() {
        var accumulator = TemporalAccumulator(width: 4, height: 2)
        let frame: [Float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        for _ in 0..<10 {
            accumulator.addFrame(frame)
        }
        let metrics = accumulator.computeMetrics()

        #expect(metrics.frameCount == 10)
        #expect(metrics.globalMeanStdDev == 0)
        #expect(metrics.edgeBandMaxStdDev == 0)
        #expect(metrics.globalMeanAlpha == 0.5)
        #expect(metrics.edgeBandPixelPercentage == 100)
    }

    @Test func twoLevelSequenceMatchesAnalyticalStdDev() {
        // Alternating pattern: each pixel toggles between 0.4 and 0.6.
        // Mean = 0.5, variance = 0.01, σ = 0.1 exactly.
        var accumulator = TemporalAccumulator(width: 2, height: 1)
        for offset in 0..<200 {
            let value: Float = (offset % 2 == 0) ? 0.4 : 0.6
            accumulator.addFrame([value, value])
        }
        let metrics = accumulator.computeMetrics()

        #expect(abs(metrics.globalMeanStdDev - 0.1) < 1e-6)
        #expect(abs(metrics.edgeBandMeanStdDev - 0.1) < 1e-6)
        #expect(abs(metrics.globalMeanAlpha - 0.5) < 1e-6)
    }

    @Test func interiorPixelsExcludedFromEdgeBandStats() {
        // 4 pixels: two interior (α=1), two edge (oscillating). The edge
        // band stats must see only the two oscillating pixels.
        var accumulator = TemporalAccumulator(width: 4, height: 1)
        for offset in 0..<100 {
            let edgeA: Float = (offset % 2 == 0) ? 0.3 : 0.5
            let edgeB: Float = (offset % 2 == 0) ? 0.7 : 0.5
            accumulator.addFrame([1.0, edgeA, edgeB, 1.0])
        }
        let metrics = accumulator.computeMetrics()

        #expect(metrics.edgeBandPixelPercentage == 50)
        // Both edge pixels toggle by ±0.1 around 0.4 and 0.6 — analytical
        // σ is 0.1 in both cases.
        #expect(abs(metrics.edgeBandMeanStdDev - 0.1) < 1e-6)
        #expect(abs(metrics.edgeBandMaxStdDev - 0.1) < 1e-6)
    }

    @Test func emptyAccumulatorReturnsZeros() {
        let accumulator = TemporalAccumulator(width: 10, height: 10)
        let metrics = accumulator.computeMetrics()

        #expect(metrics.frameCount == 0)
        #expect(metrics.globalMeanStdDev == 0)
        #expect(metrics.edgeBandPixelPercentage == 0)
        #expect(metrics.width == 10)
        #expect(metrics.height == 10)
    }

    @Test func sortedP95IsStableAcrossInputOrder() {
        // Build a deterministic edge-band pattern with a heavy tail so p95
        // picks a specific pixel; the reported value must match regardless
        // of which order we processed the frames.
        var ascendingAccumulator = TemporalAccumulator(width: 100, height: 1)
        var descendingAccumulator = TemporalAccumulator(width: 100, height: 1)

        let frameCount = 32
        for frameIndex in 0..<frameCount {
            var frame = [Float](repeating: 0.5, count: 100)
            // 100 pixels with σ scaling linearly from 0.01 at pixel 0 to
            // 0.10 at pixel 99. Alternating +/- sign per frame flips at
            // the same cadence so mean stays at 0.5.
            let sign: Float = (frameIndex % 2 == 0) ? 1 : -1
            for pixel in 0..<100 {
                let amplitude = Float(pixel + 1) * 0.001
                frame[pixel] = 0.5 + sign * amplitude
            }
            ascendingAccumulator.addFrame(frame)
            descendingAccumulator.addFrame(frame.reversed())
        }
        let ascendingMetrics = ascendingAccumulator.computeMetrics()
        let descendingMetrics = descendingAccumulator.computeMetrics()

        #expect(abs(ascendingMetrics.edgeBandP95StdDev - descendingMetrics.edgeBandP95StdDev) < 1e-12)
        #expect(abs(ascendingMetrics.edgeBandMaxStdDev - descendingMetrics.edgeBandMaxStdDev) < 1e-12)
    }
}
