//
//  TemporalBlendTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Golden tests for the Phase 1 temporal-blend compute kernel. The kernel
//  blends the current frame's alpha toward the previous frame's alpha on
//  pixels where the source RGB has barely changed, so each test here drives
//  a pair of (matte, source) frames with a known stationary/moving mix and
//  asserts the expected blend ratio per pixel.
//
//  Every test runs on a real Metal device via `TestHarness`; CI machines
//  without a GPU throw `MetalUnavailable` and the test case early-returns
//  (see the `XCTSkipError` pattern used across this suite).
//

import Foundation
import Metal
import Testing
import simd
@testable import CorridorKeyToolboxMetalStages

@Suite("TemporalBlend")
struct TemporalBlendTests {

    // MARK: - Stationary pixel — should blend fully toward previous alpha

    @Test("Stationary pixel at strength 1.0 replaces current alpha with previous alpha")
    func stationaryPixelReplacesWithPrevious() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Identical source RGB in both frames → motion is zero → the gate
        // is fully open → `effectiveStrength == strength`.
        let source = SIMD4<Float>(0.5, 0.5, 0.5, 1.0)
        let currentMatte = SIMD4<Float>(0.9, 0, 0, 1.0)
        let previousMatte = SIMD4<Float>(0.1, 0, 0, 1.0)

        let output = try runBlend(
            entry: entry,
            currentMatte: currentMatte,
            previousMatte: previousMatte,
            currentSource: source,
            previousSource: source,
            strength: 1.0,
            motionThreshold: 0.05
        )

        // strength 1.0 + motion 0 → output == previousMatte
        #expect(abs(output.x - previousMatte.x) < 1e-4, "Expected full replacement to previous, got \(output.x)")
    }

    @Test("Stationary pixel at strength 0.5 averages current and previous alpha")
    func stationaryPixelHalfStrengthAverages() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let source = SIMD4<Float>(0.25, 0.5, 0.75, 1.0)
        let currentMatte = SIMD4<Float>(0.8, 0, 0, 1.0)
        let previousMatte = SIMD4<Float>(0.4, 0, 0, 1.0)
        let strength: Float = 0.5

        let output = try runBlend(
            entry: entry,
            currentMatte: currentMatte,
            previousMatte: previousMatte,
            currentSource: source,
            previousSource: source,
            strength: strength,
            motionThreshold: 0.05
        )

        // α = current + (previous − current) × strength = 0.8 − 0.2 = 0.6
        let expected = currentMatte.x + (previousMatte.x - currentMatte.x) * strength
        #expect(abs(output.x - expected) < 1e-4, "Expected \(expected), got \(output.x)")
    }

    // MARK: - Motion — should pass current alpha through unchanged

    @Test("Large motion passes current alpha through unchanged")
    func largeMotionBypassesBlend() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // RGB residual of 0.5 on the R channel is well above `2 × 0.05`, so
        // the gate collapses to zero regardless of strength.
        let currentSource = SIMD4<Float>(0.9, 0.5, 0.5, 1.0)
        let previousSource = SIMD4<Float>(0.2, 0.5, 0.5, 1.0)
        let currentMatte = SIMD4<Float>(0.75, 0, 0, 1.0)
        let previousMatte = SIMD4<Float>(0.1, 0, 0, 1.0)

        let output = try runBlend(
            entry: entry,
            currentMatte: currentMatte,
            previousMatte: previousMatte,
            currentSource: currentSource,
            previousSource: previousSource,
            strength: 1.0,
            motionThreshold: 0.05
        )

        // Motion is 0.7 ≫ 2 × 0.05 → gate 0 → α unchanged.
        #expect(abs(output.x - currentMatte.x) < 1e-4, "Expected current alpha passthrough, got \(output.x)")
    }

    @Test("Motion exactly at threshold halves the effective blend strength")
    func motionAtThresholdHalvesStrength() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // motion == threshold → gate = 1 − threshold/(2·threshold) = 0.5.
        let motionDelta: Float = 0.05
        let currentSource = SIMD4<Float>(0.5, 0.5, 0.5, 1.0)
        let previousSource = SIMD4<Float>(0.5 + motionDelta, 0.5, 0.5, 1.0)
        let currentMatte = SIMD4<Float>(1.0, 0, 0, 1.0)
        let previousMatte = SIMD4<Float>(0.0, 0, 0, 1.0)
        let strength: Float = 1.0

        let output = try runBlend(
            entry: entry,
            currentMatte: currentMatte,
            previousMatte: previousMatte,
            currentSource: currentSource,
            previousSource: previousSource,
            strength: strength,
            motionThreshold: motionDelta
        )

        // Effective strength = 1.0 × 0.5 = 0.5 → α = 1.0 + (0.0 − 1.0)·0.5 = 0.5
        #expect(abs(output.x - 0.5) < 1e-4, "Expected 0.5, got \(output.x)")
    }

    // MARK: - Gate edges

    @Test("Max-channel motion gate uses the largest of the three RGB deltas")
    func motionGateUsesMaxChannel() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Two channels stationary, blue channel jumps by 0.3. Motion metric
        // is `max`, so the gate must read 0.3 — well above `2 × 0.05`.
        let currentSource = SIMD4<Float>(0.5, 0.5, 0.5, 1.0)
        let previousSource = SIMD4<Float>(0.5, 0.5, 0.8, 1.0)
        let currentMatte = SIMD4<Float>(0.8, 0, 0, 1.0)
        let previousMatte = SIMD4<Float>(0.2, 0, 0, 1.0)

        let output = try runBlend(
            entry: entry,
            currentMatte: currentMatte,
            previousMatte: previousMatte,
            currentSource: currentSource,
            previousSource: previousSource,
            strength: 1.0,
            motionThreshold: 0.05
        )

        // Gate collapses to 0 → α == currentMatte.x.
        #expect(abs(output.x - currentMatte.x) < 1e-4, "Expected current passthrough, got \(output.x)")
    }

    @Test("applyTemporalBlend returns nil when strength is zero")
    func zeroStrengthReturnsNil() throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let source = SIMD4<Float>(0.5, 0.5, 0.5, 1.0)
        let matte = SIMD4<Float>(0.5, 0, 0, 1.0)
        let currentMatteTexture = try makeRGBATexture(entry: entry, width: 2, height: 2, colour: matte)
        let previousMatteTexture = try makeRGBATexture(entry: entry, width: 2, height: 2, colour: matte)
        let sourceTexture = try makeRGBATexture(entry: entry, width: 2, height: 2, colour: source)

        guard let queue = entry.device.makeCommandQueue() else {
            throw MetalUnavailable(reason: "Could not create command queue.")
        }
        guard let commandBuffer = queue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }

        let result = try RenderStages.applyTemporalBlend(
            currentMatte: currentMatteTexture,
            previousMatte: previousMatteTexture,
            currentSource: sourceTexture,
            previousSource: sourceTexture,
            strength: 0,
            motionThreshold: 0.05,
            entry: entry,
            commandBuffer: commandBuffer
        )
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()

        #expect(result == nil, "Expected nil when strength is zero; got a pooled texture")
    }

    // MARK: - Helpers

    /// Encodes the temporal-blend stage into a fresh command buffer with
    /// four solid-colour 2×2 input textures and returns the first pixel of
    /// the resulting matte. Uses RGBA32Float throughout for deterministic
    /// read-back — the kernel writes `float4(alpha, 0, 0, 1)` so only the
    /// R channel carries the matte value we assert on.
    private func runBlend(
        entry: MetalDeviceCacheEntry,
        currentMatte: SIMD4<Float>,
        previousMatte: SIMD4<Float>,
        currentSource: SIMD4<Float>,
        previousSource: SIMD4<Float>,
        strength: Float,
        motionThreshold: Float
    ) throws -> SIMD4<Float> {
        let currentMatteTex = try makeRGBATexture(entry: entry, width: 2, height: 2, colour: currentMatte)
        let previousMatteTex = try makeRGBATexture(entry: entry, width: 2, height: 2, colour: previousMatte)
        let currentSourceTex = try makeRGBATexture(entry: entry, width: 2, height: 2, colour: currentSource)
        let previousSourceTex = try makeRGBATexture(entry: entry, width: 2, height: 2, colour: previousSource)

        guard let queue = entry.device.makeCommandQueue() else {
            throw MetalUnavailable(reason: "Could not create command queue.")
        }
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalUnavailable(reason: "Could not create command buffer.")
        }

        let pooled = try RenderStages.applyTemporalBlend(
            currentMatte: currentMatteTex,
            previousMatte: previousMatteTex,
            currentSource: currentSourceTex,
            previousSource: previousSourceTex,
            strength: strength,
            motionThreshold: motionThreshold,
            entry: entry,
            commandBuffer: commandBuffer
        )
        guard let pooled else {
            throw MetalUnavailable(reason: "Temporal blend unexpectedly returned nil.")
        }

        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()
        if let error = commandBuffer.error { throw error }

        var pixel = SIMD4<Float>(0, 0, 0, 0)
        pooled.texture.getBytes(
            &pixel,
            bytesPerRow: pooled.texture.width * MemoryLayout<SIMD4<Float>>.stride,
            from: MTLRegionMake2D(0, 0, 1, 1),
            mipmapLevel: 0
        )
        pooled.returnManually()
        return pixel
    }

    /// Uploads a 2×2 (or any size) RGBA32Float texture filled with `colour`.
    /// Shared storage so the CPU fill reaches the GPU without an explicit
    /// blit. We allocate directly via the device instead of the pool here
    /// because this test concurrently holds four input textures at once and
    /// `PooledTexture.deinit` would immediately return each to the pool,
    /// handing the next `acquire` the same physical texture and smashing
    /// the earlier fills.
    private func makeRGBATexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        colour: SIMD4<Float>
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not allocate colour texture.")
        }
        let pixels = [SIMD4<Float>](repeating: colour, count: width * height)
        pixels.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<SIMD4<Float>>.stride
                )
            }
        }
        return texture
    }
}
