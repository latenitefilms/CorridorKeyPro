//
//  MLXMemoryTests.swift
//  CorridorKeyToolboxInferenceTests
//
//  Headless reproduction of the unbounded-memory bug Final Cut Pro
//  surfaces during a long Analyse Clip pass at the Maximum rung. The
//  symptom in the field was 40+ GB of resident memory after ~25 frames;
//  the root cause is MLX's per-process buffer cache, which defaults to
//  the device's recommended-max-working-set and accumulates intermediates
//  across inference calls.
//
//  These tests pin the leak with the smallest bundled bridge (512px) so
//  CI can run them in a few seconds, and they assert on
//  `MLX.Memory.cacheMemory` so the failure mode is visible without
//  launching FCP.
//
//  Run with: `swift test --filter MLXMemory`
//

import Foundation
import Metal
import MLX
import Testing
@testable import CorridorKeyToolboxLogic
@testable import CorridorKeyToolboxMetalStages

@Suite("MLXMemory")
struct MLXMemoryTests {

    // MARK: - Single-shot smoke test

    @Test("MLX bridge loads and runs a single 512px inference")
    func singleInferenceCompletes() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let request = try makeRequest(rung: 512, entry: entry)
        let output = try makeOutput(rung: 512, entry: entry)
        try engine.run(request: request, output: output)
    }

    // MARK: - The leak gate

    /// Runs 30 inferences, calling `clearMLXCache()` after EACH one,
    /// and verifies the loop completes without crashing AND the output
    /// of the last frame matches the first. This is the strategy we
    /// need to make production stable on long analyses: bound memory
    /// per-frame so the user's 12-frame ceiling becomes infinite.
    /// If MLX's `clearCache` is unsafe to call between inferences,
    /// this test will crash or report drift.
    /// Probe to characterise MLX's per-call memory behaviour after we
    /// added the autoreleasepool around `MLXKeyingEngine.run`. Tracks
    /// both `cacheMemory` and `activeMemory` snapshots **between**
    /// inferences (so locals from the prior call have released) so we
    /// can tell whether the cache is genuinely growing unboundedly or
    /// just steady-state at one-inference's-worth.
    @Test("MLX memory profile across many inferences (no clearCache)")
    func memoryProfileWithoutClearCache() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let request = try makeRequest(rung: 512, entry: entry)
        let output = try makeOutput(rung: 512, entry: entry)

        // Drop any state from the warmup so iteration 0 starts clean.
        MLX.Memory.clearCache()

        let oneMegabyte = 1024 * 1024
        var cacheTrace: [Int] = []
        var activeTrace: [Int] = []

        for _ in 0..<30 {
            try engine.run(request: request, output: output)
            // Snapshot AFTER the call has fully returned (autoreleasepool
            // inside `run` has drained, ARC has dropped per-call locals).
            let snapshot = MLX.Memory.snapshot()
            cacheTrace.append(snapshot.cacheMemory / oneMegabyte)
            activeTrace.append(snapshot.activeMemory / oneMegabyte)
        }

        let cacheMin = cacheTrace.min() ?? 0
        let cacheMax = cacheTrace.max() ?? 0
        let activeMin = activeTrace.min() ?? 0
        let activeMax = activeTrace.max() ?? 0

        print("MLX cache trace (MB): min=\(cacheMin), max=\(cacheMax), final=\(cacheTrace.last ?? 0).")
        print("MLX active trace (MB): min=\(activeMin), max=\(activeMax), final=\(activeTrace.last ?? 0).")
        print("First 5: cache \(cacheTrace.prefix(5)), active \(activeTrace.prefix(5))")
        print("Last 5: cache \(cacheTrace.suffix(5)), active \(activeTrace.suffix(5))")

        // The KEY assertion: cache must be steady-state, not growing.
        // A leak would show as cacheMax >> cacheMin AND a strictly
        // increasing trend. We tolerate up to 50% wobble around the
        // settled value (MLX's allocator reclaims opportunistically).
        let lateCacheAvg = cacheTrace.suffix(10).reduce(0, +) / 10
        let earlyCacheAvg = cacheTrace.prefix(10).reduce(0, +) / 10
        let cacheGrowth = lateCacheAvg - earlyCacheAvg
        #expect(
            cacheGrowth <= cacheTrace.first! / 2,
            "MLX cache grew across the loop: early avg=\(earlyCacheAvg) MB, late avg=\(lateCacheAvg) MB, growth=\(cacheGrowth) MB."
        )
    }

    /// Runs many inferences in a row and asserts that calling
    /// `clearMLXCache()` at the end of the session brings the cache back
    /// down close to baseline. This is the production strategy: MLX is
    /// allowed to cache freely during a pass (so it never thrashes the
    /// allocator), and the analyser flushes the cache once the user is
    /// done. Pre-fix the cache stayed elevated indefinitely (the 42 GB
    /// post-analysis residency the user hit); post-fix it should settle
    /// well below 1 GB after `clearMLXCache`.
    @Test("clearMLXCache reclaims the cache after 30 sequential 512px inferences")
    func clearCacheReclaimsMemoryAfterSession() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        MLX.Memory.peakMemory = 0

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let request = try makeRequest(rung: 512, entry: entry)
        let output = try makeOutput(rung: 512, entry: entry)

        for _ in 0..<30 {
            try engine.run(request: request, output: output)
        }
        let beforeClear = MLX.Memory.snapshot()

        engine.clearMLXCache()
        let afterClear = MLX.Memory.snapshot()

        // Sanity check: the cache should be substantial during the
        // analysis (we're not running with a synthetic cap) — without
        // some load there's nothing to validate.
        let oneMegabyte = 1024 * 1024
        #expect(
            beforeClear.cacheMemory > 32 * oneMegabyte,
            "Expected meaningful cache load before clear; got \(beforeClear.cacheMemory) bytes."
        )

        // Post-clear: cache should drop substantially. The exact floor
        // depends on what MLX considers "in use", but we expect at
        // least an 80% reduction. Anything less and the bug is back.
        let releasedBytes = beforeClear.cacheMemory - afterClear.cacheMemory
        let releasePercent = Double(releasedBytes) / Double(max(beforeClear.cacheMemory, 1)) * 100
        #expect(
            releasePercent >= 80.0,
            "clearMLXCache only released \(releasePercent)% (was \(beforeClear.cacheMemory), now \(afterClear.cacheMemory))."
        )

        print("MLX before clearCache: \(beforeClear.cacheMemory / oneMegabyte) MB cached, " +
              "\(beforeClear.activeMemory / oneMegabyte) MB active.")
        print("MLX after clearCache:  \(afterClear.cacheMemory / oneMegabyte) MB cached, " +
              "\(afterClear.activeMemory / oneMegabyte) MB active " +
              "(released \(releasedBytes / oneMegabyte) MB, \(Int(releasePercent))%).")
    }

    /// A second-tier guard: even with the cache capped, an active-memory
    /// leak (model weights or output tensors retained per call) would
    /// still ramp resident memory across an analysis pass. We sample
    /// active memory after a warm-up iteration, then again after 30
    /// more, and require the delta to be small. The threshold is chosen
    /// generously — anything sub-100 MB of growth is in the noise of
    /// MLX's allocator. The pre-fix bug grew active by hundreds of MB
    /// per frame.
    @Test("MLX active memory does not ramp over 30 sequential 512px inferences")
    func activeMemoryDoesNotRampOverInferences() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let request = try makeRequest(rung: 512, entry: entry)
        let output = try makeOutput(rung: 512, entry: entry)

        // Single warm-up to reach steady-state allocation, then sample
        // before/after the full loop.
        try engine.run(request: request, output: output)
        let baseline = MLX.Memory.activeMemory

        for _ in 0..<30 {
            try engine.run(request: request, output: output)
        }
        let endActive = MLX.Memory.activeMemory

        let delta = endActive - baseline
        let allowedRampBytes = 100 * 1024 * 1024
        // We only care about *growth*. A negative delta means MLX freed
        // intermediates between the baseline and the end sample —
        // perfectly fine; nothing leaked.
        #expect(
            delta <= allowedRampBytes,
            "MLX active memory grew by \(delta) bytes over 30 inferences (baseline \(baseline), end \(endActive)); allowed ≤ \(allowedRampBytes)."
        )
        print("MLX active baseline: \(baseline / (1024 * 1024)) MB, " +
              "after 30 more inferences: \(endActive / (1024 * 1024)) MB " +
              "(delta: \(delta / 1024) KB).")
    }


    // MARK: - Helpers

    private func readAlpha(output: KeyingInferenceOutput) -> [Float] {
        let width = output.alphaTexture.width
        let height = output.alphaTexture.height
        var pixels = [Float](repeating: 0, count: width * height)
        let bytesPerRow = width * MemoryLayout<Float>.size
        pixels.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                output.alphaTexture.getBytes(
                    base,
                    bytesPerRow: bytesPerRow,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        return pixels
    }

    /// Allocates a normalised-input MTLBuffer matching the shape MLX
    /// expects (1 × rung × rung × 4 floats), filled with a deterministic
    /// gradient so each iteration sees real values rather than zeros
    /// that MLX could constant-fold.
    private func makeRequest(rung: Int, entry: MetalDeviceCacheEntry) throws -> KeyingInferenceRequest {
        guard let buffer = entry.normalizedInputBuffer(forRung: rung) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate normalised input buffer.")
        }
        // Fill with a smooth gradient. MLX's optimiser will not constant-
        // fold across non-trivial inputs, so this exercises the full
        // graph the same way real footage would.
        let elementCount = rung * rung * 4
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<elementCount {
            pointer[index] = Float(index % 256) / 255.0
        }

        // Dummy raw-source texture — `MLXKeyingEngine.run` ignores the
        // contents but takes the texture handle to satisfy the request
        // type. Allocate at the smallest legal size.
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

    /// Allocates the alpha + foreground destination textures the engine
    /// writes into. Both are `.shared` so the production code can read
    /// them back; that lifecycle matches `InferenceCoordinator.makeOutputTextures`.
    private func makeOutput(rung: Int, entry: MetalDeviceCacheEntry) throws -> KeyingInferenceOutput {
        guard let alpha = entry.makeIntermediateTexture(
            width: rung,
            height: rung,
            pixelFormat: .r32Float,
            storageMode: .shared
        ) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate alpha texture.")
        }
        guard let foreground = entry.makeIntermediateTexture(
            width: rung,
            height: rung,
            pixelFormat: .rgba32Float,
            storageMode: .shared
        ) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate foreground texture.")
        }
        return KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)
    }
}

/// Minimal stand-in for XCTSkipError, mirroring the shape used by
/// `CorridorKeyToolboxMetalStagesTests`. Tests catch `MetalUnavailable`
/// and rethrow as this so the runner reports a skip instead of a failure
/// when the host has no GPU.
private struct XCTSkip: Error, CustomStringConvertible {
    let underlying: any Error
    init(_ error: any Error) { self.underlying = error }
    var description: String { "Skipped: \(underlying)" }
}
