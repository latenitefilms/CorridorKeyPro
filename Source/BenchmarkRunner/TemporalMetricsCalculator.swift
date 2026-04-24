//
//  TemporalMetricsCalculator.swift
//  BenchmarkRunner
//
//  Computes per-pixel temporal variance over a matte sequence and derives
//  aggregate metrics that quantify "edge flicker" — the #1 visual tell of
//  AI-generated mattes.
//
//  Designed as two layers: an `TemporalAccumulator` value type owning the
//  pure running-stats math so tests can exercise it with synthetic arrays,
//  and `TemporalMetricsCalculator.compute(from:)` which composes it with
//  the disk I/O path. Variance is derived via the `E[X²] - E[X]²` identity
//  so a second pass over the data isn't needed.
//

import Foundation

struct TemporalMetrics: Sendable, Equatable {
    /// Mean per-pixel standard deviation over the whole frame.
    let globalMeanStdDev: Double
    /// Mean σ restricted to pixels whose mean alpha falls in the edge band
    /// (0.05 ≤ mean α ≤ 0.95). This is the number editors notice — interior
    /// pixels read as 1.0 and exterior as 0.0, so they do not flicker.
    let edgeBandMeanStdDev: Double
    /// Percentage of pixels that qualify as edge-band (useful for context).
    let edgeBandPixelPercentage: Double
    /// p95 / max σ, both over edge-band pixels only. Interior pixels dwarf
    /// these stats at p100 because they are trivially stable at σ ≈ 0.
    let edgeBandP95StdDev: Double
    let edgeBandMaxStdDev: Double
    /// Mean of mean-alpha over all pixels — a sanity check that the mattes
    /// haven't inverted or been loaded from the wrong channel.
    let globalMeanAlpha: Double
    /// Number of frames that contributed to the statistics.
    let frameCount: Int
    /// Dimensions of the matte (must be consistent across the sequence).
    let width: Int
    let height: Int
}

/// Streaming statistics over a matte sequence. Accumulates `Σx` and `Σx²`
/// per pixel so neither the caller nor the accumulator needs to hold every
/// frame in memory. One instance per sequence; pass each frame through
/// `addFrame` in order, then call `computeMetrics` once at the end.
struct TemporalAccumulator {
    let width: Int
    let height: Int
    private var runningSum: [Double]
    private var runningSumOfSquares: [Double]
    private(set) var frameCount: Int

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.runningSum = [Double](repeating: 0, count: width * height)
        self.runningSumOfSquares = [Double](repeating: 0, count: width * height)
        self.frameCount = 0
    }

    /// Pass a flat top-to-bottom row-major frame. Values are expected in
    /// the `[0, 1]` matte range; values outside that range are accepted but
    /// will inflate variance.
    mutating func addFrame(_ pixels: [Float]) {
        precondition(pixels.count == runningSum.count, "Frame size mismatch")
        for index in 0..<pixels.count {
            let value = Double(pixels[index])
            runningSum[index] += value
            runningSumOfSquares[index] += value * value
        }
        frameCount += 1
    }

    /// Reduces the running buffers into the aggregate report. Safe to call
    /// multiple times; each call re-computes from the accumulators.
    func computeMetrics() -> TemporalMetrics {
        guard frameCount > 0 else {
            return TemporalMetrics(
                globalMeanStdDev: 0,
                edgeBandMeanStdDev: 0,
                edgeBandPixelPercentage: 0,
                edgeBandP95StdDev: 0,
                edgeBandMaxStdDev: 0,
                globalMeanAlpha: 0,
                frameCount: 0,
                width: width,
                height: height
            )
        }

        let divisor = Double(frameCount)
        let pixelCount = runningSum.count

        var totalStdDev = 0.0
        var totalMeanAlpha = 0.0
        var edgeBandStdDevSum = 0.0
        var edgeBandStdDevValues: [Double] = []
        edgeBandStdDevValues.reserveCapacity(pixelCount / 4)
        var edgeBandMaxStdDev = 0.0

        for pixelIndex in 0..<pixelCount {
            let meanAlpha = runningSum[pixelIndex] / divisor
            let meanSquared = runningSumOfSquares[pixelIndex] / divisor
            // Floating-point drift can push variance microscopically
            // negative when the signal is effectively constant. Clamp so
            // `sqrt` stays in the reals.
            let variance = max(0, meanSquared - meanAlpha * meanAlpha)
            let stdDev = variance.squareRoot()

            totalStdDev += stdDev
            totalMeanAlpha += meanAlpha

            if meanAlpha >= 0.05 && meanAlpha <= 0.95 {
                edgeBandStdDevSum += stdDev
                edgeBandStdDevValues.append(stdDev)
                if stdDev > edgeBandMaxStdDev {
                    edgeBandMaxStdDev = stdDev
                }
            }
        }

        let pixelCountDouble = Double(pixelCount)
        let globalMeanStdDev = totalStdDev / pixelCountDouble
        let globalMeanAlpha = totalMeanAlpha / pixelCountDouble

        let edgeBandMeanStdDev: Double
        let edgeBandP95StdDev: Double
        let edgeBandPercentage: Double
        if edgeBandStdDevValues.isEmpty {
            edgeBandMeanStdDev = 0
            edgeBandP95StdDev = 0
            edgeBandPercentage = 0
        } else {
            edgeBandMeanStdDev = edgeBandStdDevSum / Double(edgeBandStdDevValues.count)
            edgeBandPercentage = 100.0 * Double(edgeBandStdDevValues.count) / pixelCountDouble

            edgeBandStdDevValues.sort()
            let p95Index = min(
                edgeBandStdDevValues.count - 1,
                Int((Double(edgeBandStdDevValues.count) * 0.95).rounded(.down))
            )
            edgeBandP95StdDev = edgeBandStdDevValues[p95Index]
        }

        return TemporalMetrics(
            globalMeanStdDev: globalMeanStdDev,
            edgeBandMeanStdDev: edgeBandMeanStdDev,
            edgeBandPixelPercentage: edgeBandPercentage,
            edgeBandP95StdDev: edgeBandP95StdDev,
            edgeBandMaxStdDev: edgeBandMaxStdDev,
            globalMeanAlpha: globalMeanAlpha,
            frameCount: frameCount,
            width: width,
            height: height
        )
    }
}

enum TemporalMetricsCalculator {

    /// Streams every frame of `sequence` through the accumulator and
    /// returns the aggregate metrics. Progress is logged every 50 frames
    /// so long runs stay interactive.
    static func compute(from sequence: MatteSequence) async throws -> TemporalMetrics {
        guard let firstFrame = sequence.frames.first else {
            throw BenchmarkError.emptyFolder(sequence.folder)
        }

        var accumulator = TemporalAccumulator(
            width: firstFrame.width,
            height: firstFrame.height
        )

        let frameCount = sequence.frames.count
        for (offset, frame) in sequence.frames.enumerated() {
            let buffer = try MatteImageLoader.load(frame.url)
            guard buffer.width == accumulator.width, buffer.height == accumulator.height else {
                throw BenchmarkError.imageLoadFailed(
                    frame.url,
                    underlying: MismatchError(
                        expected: (accumulator.width, accumulator.height),
                        got: (buffer.width, buffer.height)
                    )
                )
            }
            accumulator.addFrame(buffer.pixels)

            if offset == 0 || (offset + 1) % 50 == 0 || offset == frameCount - 1 {
                print("  frame \(offset + 1)/\(frameCount)")
            }
        }

        return accumulator.computeMetrics()
    }

    private struct MismatchError: Error, CustomStringConvertible {
        let expected: (Int, Int)
        let got: (Int, Int)
        var description: String {
            "Frame dimensions mismatch — expected \(expected.0)x\(expected.1), got \(got.0)x\(got.1)"
        }
    }
}
