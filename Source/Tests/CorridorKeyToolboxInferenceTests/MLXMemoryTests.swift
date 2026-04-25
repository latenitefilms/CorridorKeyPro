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

    /// Runs many inferences in a row and asserts MLX's cache stays bounded.
    /// Without the cache-limit cap, on a 32 GB M1 Max the cache balloons
    /// past several GB after a handful of 2048 frames; without the fix at
    /// 512 we measured 4.4 GB of cached buffers after only 30 iterations.
    /// With the fix in `MLXKeyingEngine.applyMemoryLimitsOnce` the cache
    /// settles at ~260 MB (the MLX cache limit + a few MB of overshoot
    /// while the allocator decides what to evict). The thresholds below
    /// are deliberately generous so the test isn't flaky on different
    /// hardware — we're guarding against the unbounded-growth bug, not
    /// micro-optimising MLX's allocator. Anything an order of magnitude
    /// larger than the limit means the fix has regressed.
    @Test("MLX cache stays bounded across 30 sequential 512px inferences")
    func cacheStaysBoundedAcrossThirtyInferences() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        // Reset the peak counter so we measure only this test run.
        MLX.Memory.peakMemory = 0

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let request = try makeRequest(rung: 512, entry: entry)
        let output = try makeOutput(rung: 512, entry: entry)

        // The cache limit is 256 MB; allow a 64 MB overshoot for MLX's
        // lazy reclaim (it triggers eviction "after" exceeding, not at).
        // Anything ≥ 1 GB indicates the fix has regressed.
        let cacheCeilingBytes = (256 + 64) * 1024 * 1024
        var observedPeakCacheBytes = 0

        for iteration in 0..<30 {
            try engine.run(request: request, output: output)
            // Touch every iteration so the cache pattern matches the
            // real analyse loop — InferenceCoordinator's `cachedMLXOutput`
            // is replaced each frame in production, which drops the
            // previous output references.

            let snapshot = MLX.Memory.snapshot()
            observedPeakCacheBytes = max(observedPeakCacheBytes, snapshot.cacheMemory)

            #expect(
                snapshot.cacheMemory <= cacheCeilingBytes,
                "MLX cache grew past \(cacheCeilingBytes) bytes at iteration \(iteration): \(snapshot)"
            )
        }

        let snapshot = MLX.Memory.snapshot()
        print("MLX after 30 inferences — peak: \(snapshot.peakMemory / (1024 * 1024)) MB, " +
              "active: \(snapshot.activeMemory / (1024 * 1024)) MB, " +
              "cache: \(snapshot.cacheMemory / (1024 * 1024)) MB " +
              "(observed peak cache: \(observedPeakCacheBytes / (1024 * 1024)) MB).")
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
