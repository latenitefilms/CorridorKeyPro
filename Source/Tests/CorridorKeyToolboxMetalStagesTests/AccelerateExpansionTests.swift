//
//  AccelerateExpansionTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Phase 3.2 replaced a per-pixel scalar loop in `MLXKeyingEngine.
//  writeForegroundBuffer` with three strided `cblas_scopy` calls plus a
//  `vDSP_vfill` for the alpha stride. This test pins the correctness of
//  that Accelerate path by running both paths against the same synthetic
//  input and asserting bitwise-identical output.
//
//  Correctness is the only thing we verify here — we don't care about the
//  speed-up in CI (the intention is measuring 10× via Instruments, not a
//  wall-time assertion). Bitwise identity is a stronger guarantee than a
//  tolerance comparison and catches index arithmetic regressions.
//

import Foundation
import Accelerate
import Testing

@Suite("AccelerateExpansion")
struct AccelerateExpansionTests {

    @Test("cblas_scopy interleave matches the scalar loop at 1024²")
    func interleaveMatches1K() {
        verifyInterleave(pixelCount: 1024 * 1024)
    }

    @Test("cblas_scopy interleave matches the scalar loop at 2048²")
    func interleaveMatches2K() {
        verifyInterleave(pixelCount: 2048 * 2048)
    }

    // MARK: - Helpers

    /// Synthesises a tightly-packed RGB float buffer, runs both expansion
    /// paths, and checks that every output word is bit-identical.
    private func verifyInterleave(pixelCount: Int) {
        let rgbBuffer = makeDeterministicRGB(pixelCount: pixelCount)

        let scalarOutput = expandWithScalarLoop(rgb: rgbBuffer, pixelCount: pixelCount)
        let acceleratedOutput = expandWithAccelerate(rgb: rgbBuffer, pixelCount: pixelCount)

        #expect(scalarOutput.count == acceleratedOutput.count)
        for index in 0..<acceleratedOutput.count where scalarOutput[index] != acceleratedOutput[index] {
            Issue.record("Mismatch at index \(index): scalar=\(scalarOutput[index]) accel=\(acceleratedOutput[index])")
            return
        }
    }

    private func makeDeterministicRGB(pixelCount: Int) -> [Float] {
        var buffer = [Float](repeating: 0, count: pixelCount * 3)
        for index in 0..<pixelCount {
            // Deterministic pseudo-random pattern so we don't depend on
            // `Random`'s platform seed. Each channel uses a different
            // frequency so the test catches index-shuffling bugs.
            buffer[index * 3 + 0] = Float((index * 13) % 1000) / 1000.0
            buffer[index * 3 + 1] = Float((index * 17) % 1000) / 1000.0
            buffer[index * 3 + 2] = Float((index * 23) % 1000) / 1000.0
        }
        return buffer
    }

    private func expandWithScalarLoop(rgb: [Float], pixelCount: Int) -> [Float] {
        var rgba = [Float](repeating: 0, count: pixelCount * 4)
        for index in 0..<pixelCount {
            rgba[index * 4 + 0] = rgb[index * 3 + 0]
            rgba[index * 4 + 1] = rgb[index * 3 + 1]
            rgba[index * 4 + 2] = rgb[index * 3 + 2]
            rgba[index * 4 + 3] = 1
        }
        return rgba
    }

    private func expandWithAccelerate(rgb: [Float], pixelCount: Int) -> [Float] {
        var rgba = [Float](repeating: 0, count: pixelCount * 4)
        rgba.withUnsafeMutableBufferPointer { rgbaPointer in
            guard let rgbaBase = rgbaPointer.baseAddress else { return }
            rgb.withUnsafeBufferPointer { sourcePointer in
                guard let sourceBase = sourcePointer.baseAddress else { return }
                let count = Int32(pixelCount)
                cblas_scopy(count, sourceBase + 0, 3, rgbaBase + 0, 4)
                cblas_scopy(count, sourceBase + 1, 3, rgbaBase + 1, 4)
                cblas_scopy(count, sourceBase + 2, 3, rgbaBase + 2, 4)
                var one: Float = 1.0
                vDSP_vfill(&one, rgbaBase + 3, 4, vDSP_Length(pixelCount))
            }
        }
        return rgba
    }
}
