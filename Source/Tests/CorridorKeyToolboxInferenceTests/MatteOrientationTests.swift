//
//  MatteOrientationTests.swift
//  CorridorKeyToolboxInferenceTests
//
//  Regression gate for the upside-down matte bug. Two flips were
//  added in separate commits (`a4190e0` Zero-copy MLX I/O on the GPU
//  writeback side, `9a0eda3` Renamed on the CPU cache-load side), and
//  having both produced a double-flip — the analysed cache rendered
//  upside down at compose time, with the model's "subject" pixels
//  spilling over the actual subject and exposing green-screen
//  background underneath.
//
//  These tests pin the contract: the alpha bytes that come back from
//  `MLXKeyingEngine.run` (after `writeAlphaBufferToTexture` flips them
//  into y-down) must already match the source's pixel orientation,
//  with row 0 = visual top. No further row-reversal anywhere
//  downstream.
//

import Foundation
import Metal
import MLX
import Testing
@testable import CorridorKeyToolboxLogic
@testable import CorridorKeyToolboxMetalStages

@Suite("MatteOrientation")
struct MatteOrientationTests {

    /// Builds a structured input where the **top half** of the source
    /// is green screen (alpha should = 0 there) and the **bottom half**
    /// holds the subject (alpha should = 1 there). Runs MLX, reads the
    /// alpha back, and checks that row 0 is the green half (low alpha)
    /// and the last row is the subject half (high alpha). If the
    /// writeback flipped or didn't flip incorrectly, this test catches
    /// it before the matte ever lands in the FCP composite.
    @Test("Top half green, bottom half subject — alpha is high at the bottom")
    func topGreenBottomSubject() async throws {
        try await runOrientationCheck(topIsSubject: false)
    }

    /// Mirror image: subject in top half, green in bottom half. Pinned
    /// in the opposite direction so a hypothetical "always-zero"
    /// failure mode can't accidentally pass both tests.
    @Test("Top half subject, bottom half green — alpha is high at the top")
    func topSubjectBottomGreen() async throws {
        try await runOrientationCheck(topIsSubject: true)
    }

    // MARK: - Helpers

    private func runOrientationCheck(topIsSubject: Bool) async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let rung = 512
        let request = try makeRequest(rung: rung, entry: entry, topIsSubject: topIsSubject)
        let output = try makeOutput(rung: rung, entry: entry)
        try engine.run(request: request, output: output)

        // Read alpha back into a [Float] in the same way
        // `extractAlphaMatteForAnalysis` does — this is exactly the
        // buffer that gets cached by FCP's custom-parameter blob and
        // later re-uploaded by `uploadCachedAlpha` for compose.
        let width = output.alphaTexture.width
        let height = output.alphaTexture.height
        var alpha = [Float](repeating: 0, count: width * height)
        let bytesPerRow = width * MemoryLayout<Float>.size
        alpha.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                output.alphaTexture.getBytes(
                    base,
                    bytesPerRow: bytesPerRow,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }

        // Pick representative samples from the visual top and bottom
        // of the buffer (in the same y-down convention compose uses).
        // Row 0 should match the top half of the synthetic input;
        // row height-1 the bottom half.
        let topRowAvg = rowAverage(alpha: alpha, width: width, row: 4)
        let bottomRowAvg = rowAverage(alpha: alpha, width: width, row: height - 5)

        if topIsSubject {
            let message = "Subject was on top; expected high alpha at top, low at bottom. Got topAvg=\(topRowAvg), bottomAvg=\(bottomRowAvg)."
            #expect(topRowAvg > bottomRowAvg + 0.2, Comment(rawValue: message))
        } else {
            let message = "Subject was at bottom; expected high alpha at bottom, low at top. Got topAvg=\(topRowAvg), bottomAvg=\(bottomRowAvg)."
            #expect(bottomRowAvg > topRowAvg + 0.2, Comment(rawValue: message))
        }
        print("orientation \(topIsSubject ? "topSubject" : "bottomSubject"): topAvg=\(topRowAvg), bottomAvg=\(bottomRowAvg)")
    }

    private func rowAverage(alpha: [Float], width: Int, row: Int) -> Float {
        let start = row * width
        var sum: Float = 0
        for column in 0..<width {
            sum += alpha[start + column]
        }
        return sum / Float(width)
    }

    private func makeRequest(
        rung: Int,
        entry: MetalDeviceCacheEntry,
        topIsSubject: Bool
    ) throws -> KeyingInferenceRequest {
        guard let buffer = entry.normalizedInputBuffer(forRung: rung) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate normalised input buffer.")
        }
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        // Two horizontal bands: green (background hint=1) on one half,
        // magenta-ish "subject" (hint=0) on the other half. The buffer
        // is laid out [1, H, W, 4] in **y-down** order — row 0 = visual
        // top. The model's output should mirror that orientation.
        for row in 0..<rung {
            let isSubjectRow: Bool
            if topIsSubject {
                isSubjectRow = row < rung / 2
            } else {
                isSubjectRow = row >= rung / 2
            }
            for column in 0..<rung {
                let baseIndex = (row * rung + column) * 4
                if isSubjectRow {
                    pointer[baseIndex + 0] = 0.85
                    pointer[baseIndex + 1] = 0.30
                    pointer[baseIndex + 2] = 0.50
                    pointer[baseIndex + 3] = 0.0
                } else {
                    pointer[baseIndex + 0] = 0.10
                    pointer[baseIndex + 1] = 0.85
                    pointer[baseIndex + 2] = 0.10
                    pointer[baseIndex + 3] = 1.0
                }
            }
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: rung,
            height: rung,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let rawSource = entry.device.makeTexture(descriptor: descriptor) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate raw source texture.")
        }
        return KeyingInferenceRequest(
            normalisedInputBuffer: buffer,
            rawSourceTexture: rawSource,
            inferenceResolution: rung
        )
    }

    private func makeOutput(rung: Int, entry: MetalDeviceCacheEntry) throws -> KeyingInferenceOutput {
        guard let alpha = entry.makeIntermediateTexture(
            width: rung, height: rung, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate alpha texture.")
        }
        guard let foreground = entry.makeIntermediateTexture(
            width: rung, height: rung, pixelFormat: .rgba32Float, storageMode: .shared
        ) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate foreground texture.")
        }
        return KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)
    }
}

private struct XCTSkip: Error, CustomStringConvertible {
    let underlying: any Error
    init(_ error: any Error) { self.underlying = error }
    var description: String { "Skipped: \(underlying)" }
}
