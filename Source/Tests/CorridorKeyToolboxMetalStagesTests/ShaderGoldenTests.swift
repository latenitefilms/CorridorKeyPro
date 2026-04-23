//
//  ShaderGoldenTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Golden tests for the Metal compute kernels. Each test feeds a small
//  synthetic texture through a specific stage and compares the GPU output
//  to an analytically-computed expected value, so a shader regression is
//  caught before it ever reaches the plug-in renderer.
//
//  We use synthetic inputs (pixel-level checkerboards, solid-colour
//  patches, and stripes) so the expected output is easy to derive in
//  Swift — no need for PNG fixtures and no risk of image-codec drift
//  obscuring a shader regression.
//

import Foundation
import Metal
import MetalPerformanceShaders
import Testing
import simd
import CorridorKeyToolboxLogic
@testable import CorridorKeyToolboxMetalStages

@Suite("ShaderGolden")
struct ShaderGoldenTests {

    // MARK: - Despill

    @Test("Despill Average method pulls green toward mean of R and B")
    func despillAverage() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Input: a single 2×2 patch of a strong green pixel on top of a
        // fleshtone-ish RGB. Two pixels are "spillful" (high G), two are
        // clean. All pixels share the same value so we can read any.
        let input = SIMD4<Float>(0.35, 0.75, 0.30, 1.0)
        let sourceTexture = try makeColourTexture(
            entry: entry, width: 2, height: 2, pixelFormat: .rgba32Float, colour: input
        )

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let pooled = try RenderStages.despill(
            foreground: sourceTexture,
            strength: 1.0,
            method: .average,
            entry: entry,
            commandBuffer: commandBuffer
        )
        guard let pooled else {
            Issue.record("Despill stage returned nil.")
            return
        }
        try commitAndWait(commandBuffer)
        let output = readFirstPixel(texture: pooled.texture)
        pooled.returnManually()

        // Expected: Average → limit = (R+B)/2, spill = max(0, G-limit),
        // G -= spill, R += spill*0.5, B += spill*0.5 (all at strength 1.0).
        let expectedLimit: Float = (input.x + input.z) * 0.5
        let expectedSpill = max(0, input.y - expectedLimit)
        let expectedR = input.x + expectedSpill * 0.5
        let expectedG = input.y - expectedSpill
        let expectedB = input.z + expectedSpill * 0.5

        #expect(abs(output.x - expectedR) < 1e-3, "R mismatch: \(output.x) vs \(expectedR)")
        #expect(abs(output.y - expectedG) < 1e-3, "G mismatch: \(output.y) vs \(expectedG)")
        #expect(abs(output.z - expectedB) < 1e-3, "B mismatch: \(output.z) vs \(expectedB)")
    }

    @Test("Despill Neutral with low R/B does not amplify noise past the 1e-3 guard")
    func despillNeutralStability() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Dark input with strong green cast — the exact case where the old
        // 1e-6 guard blew up. Our new 1e-3 guard keeps the output bounded.
        let input = SIMD4<Float>(0.004, 0.65, 0.003, 1.0)
        let sourceTexture = try makeColourTexture(
            entry: entry, width: 2, height: 2, pixelFormat: .rgba32Float, colour: input
        )

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let pooled = try RenderStages.despill(
            foreground: sourceTexture,
            strength: 1.0,
            method: .neutral,
            entry: entry,
            commandBuffer: commandBuffer
        )
        guard let pooled else {
            Issue.record("Despill stage returned nil.")
            return
        }
        try commitAndWait(commandBuffer)
        let output = readFirstPixel(texture: pooled.texture)
        pooled.returnManually()

        // Neutral despill clamps the result to [0, 1] (new in v1.0). That
        // clamp is the whole reason we stop amplifying noise.
        #expect(output.x >= 0 && output.x <= 1)
        #expect(output.y >= 0 && output.y <= 1)
        #expect(output.z >= 0 && output.z <= 1)
        // Green must have been reduced.
        #expect(output.y < input.y)
    }

    // MARK: - Alpha levels + gamma

    @Test("Alpha levels + gamma remaps pixels onto the configured range")
    func alphaLevelsGamma() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Create a 1-row ramp from 0..1 in 5 steps.
        let width = 5
        let inputRow: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
        guard let sourceTexture = entry.texturePool.acquire(
            width: width, height: 1, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate ramp source texture.")
            return
        }
        inputRow.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                sourceTexture.texture.replace(
                    region: MTLRegionMake2D(0, 0, width, 1),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<Float>.size
                )
            }
        }

        guard let destination = entry.texturePool.acquire(
            width: width, height: 1, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate destination texture.")
            return
        }

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }

        try RenderStages.applyAlphaLevelsGamma(
            source: sourceTexture.texture,
            destination: destination.texture,
            blackPoint: 0.2,
            whitePoint: 0.8,
            gamma: 1.0,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        var outputs = [Float](repeating: 0, count: width)
        outputs.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                destination.texture.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<Float>.size,
                    from: MTLRegionMake2D(0, 0, width, 1),
                    mipmapLevel: 0
                )
            }
        }
        sourceTexture.returnManually()
        destination.returnManually()

        for (index, input) in inputRow.enumerated() {
            let range: Float = 0.8 - 0.2
            let expected = min(max((input - 0.2) / range, 0), 1)
            #expect(abs(outputs[index] - expected) < 1e-3, "index \(index): got \(outputs[index]), expected \(expected)")
        }
    }

    // MARK: - Green hint

    @Test("Green hint kernel matches the reference formula")
    func greenHint() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Two colours: a bright green (should map near 0 matte) and a
        // warm fleshtone (should map near 1 matte).
        let commandQueue = try makeCommandQueue(entry: entry)

        func testColour(_ rgb: SIMD4<Float>) throws -> Float {
            let source = try makeColourTexture(
                entry: entry, width: 1, height: 1, pixelFormat: .rgba32Float, colour: rgb
            )
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                Issue.record("Could not create command buffer.")
                return -1
            }
            let pooled = try RenderStages.generateGreenHint(
                source: source,
                entry: entry,
                commandBuffer: commandBuffer
            )
            try commitAndWait(commandBuffer)
            var value: Float = 0
            pooled.texture.getBytes(
                &value,
                bytesPerRow: MemoryLayout<Float>.size * 2, // r16Float: 2 bytes per pixel
                from: MTLRegionMake2D(0, 0, 1, 1),
                mipmapLevel: 0
            )
            // r16Float: convert the low 16 bits back to Float.
            let halfBits = UInt16(truncatingIfNeeded: value.bitPattern)
            let hint = MatteCodec.halfToFloat(halfBits)
            pooled.returnManually()
            return hint
        }

        let greenMatte = try testColour(SIMD4<Float>(0.05, 0.85, 0.10, 1.0))
        let skinMatte = try testColour(SIMD4<Float>(0.60, 0.45, 0.40, 1.0))

        #expect(greenMatte < 0.1, "Green pixel should produce matte near 0; got \(greenMatte).")
        #expect(skinMatte > 0.95, "Skin pixel should produce matte near 1; got \(skinMatte).")
    }

    // MARK: - Refiner strength blend

    @Test("Refiner strength 1.0 is a no-op")
    func refinerStrengthNoOp() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let matte = try makeScalarTexture(entry: entry, width: 4, height: 4, value: 0.7)
        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let pooled = try RenderStages.applyRefinerStrength(
            matte: matte,
            strength: 1.0,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)
        #expect(pooled == nil, "Strength 1.0 must short-circuit to nil.")
    }

    // MARK: - Resample (Lanczos via MPS)

    @Test("Lanczos upscale produces a target-sized texture")
    func lanczosResample() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // A 4x4 solid colour texture upscaled 4x should remain the same
        // colour everywhere.
        let colour = SIMD4<Float>(0.4, 0.2, 0.8, 1.0)
        let source = try makeColourTexture(
            entry: entry, width: 4, height: 4, pixelFormat: .rgba16Float, colour: colour
        )
        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let pooled = try RenderStages.resample(
            source: source,
            targetWidth: 16,
            targetHeight: 16,
            pixelFormat: .rgba16Float,
            method: .lanczos,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        guard let pooled else {
            Issue.record("Lanczos resample returned nil.")
            return
        }
        #expect(pooled.texture.width == 16)
        #expect(pooled.texture.height == 16)
        pooled.returnManually()
    }

    // MARK: - Connected-components despeckle

    @Test("CC despeckle preserves large components and nukes specks")
    func connectedComponentsDespeckle() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Build a 32×32 matte that is zero except for:
        //   - a 20×20 block (large, keep)
        //   - a single 1×1 speck (tiny, nuke)
        let width = 32
        let height = 32
        var pixels = [Float](repeating: 0, count: width * height)
        for y in 2..<22 {
            for x in 2..<22 {
                pixels[y * width + x] = 1.0
            }
        }
        pixels[28 * width + 28] = 1.0

        guard let matte = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate matte texture.")
            return
        }
        pixels.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                matte.texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<Float>.size
                )
            }
        }

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let pooled = try RenderStages.applyConnectedComponentsDespeckle(
            matte: matte.texture,
            areaThreshold: 10,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        guard let pooled else {
            Issue.record("CC despeckle returned nil.")
            matte.returnManually()
            return
        }

        // Readback: allocate a matching `.shared` texture for the returned
        // despeckled matte so we can read it on CPU.
        guard let readbackTexture = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: pooled.texture.pixelFormat, storageMode: .shared
        ) else {
            Issue.record("Could not allocate readback texture.")
            return
        }
        guard let readbackCommandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create readback buffer.")
            return
        }
        if let blit = readbackCommandBuffer.makeBlitCommandEncoder() {
            blit.copy(
                from: pooled.texture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: width, height: height, depth: 1),
                to: readbackTexture.texture,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
        }
        try commitAndWait(readbackCommandBuffer)

        var resultPixels = [Float](repeating: 0, count: width * height)
        resultPixels.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                readbackTexture.texture.getBytes(
                    base,
                    bytesPerRow: width * pooled.texture.pixelFormat.bytesPerPixel,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }

        // Kept: centre of the large block should still be ~1.0.
        let centreIndex = 10 * width + 10
        #expect(resultPixels[centreIndex] > 0.9, "Centre pixel lost: \(resultPixels[centreIndex])")
        // Nuked: single-pixel speck should be 0.
        let speckIndex = 28 * width + 28
        #expect(resultPixels[speckIndex] < 0.1, "Speck not removed: \(resultPixels[speckIndex])")

        pooled.returnManually()
        readbackTexture.returnManually()
        matte.returnManually()
    }

    // MARK: - Helpers

    private func makeColourTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat,
        colour: SIMD4<Float>
    ) throws -> any MTLTexture {
        guard let pooled = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: pixelFormat,
            storageMode: .shared
        ) else {
            throw MetalUnavailable(reason: "Could not allocate colour texture.")
        }
        // Write the same pixel everywhere. Only rgba32Float has the exact
        // bit layout we assume here — callers should stick to that for
        // synthetic inputs.
        precondition(pixelFormat == .rgba32Float || pixelFormat == .rgba16Float,
                     "makeColourTexture supports RGBA float formats only.")
        let pixelCount = width * height
        if pixelFormat == .rgba32Float {
            var pixels = [SIMD4<Float>](repeating: colour, count: pixelCount)
            pixels.withUnsafeBufferPointer { pointer in
                if let base = pointer.baseAddress {
                    pooled.texture.replace(
                        region: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0,
                        withBytes: base,
                        bytesPerRow: width * MemoryLayout<SIMD4<Float>>.size
                    )
                }
            }
        } else {
            let halfValues: [UInt16] = [
                MatteCodec.floatToHalf(colour.x),
                MatteCodec.floatToHalf(colour.y),
                MatteCodec.floatToHalf(colour.z),
                MatteCodec.floatToHalf(colour.w)
            ]
            var pixels = [UInt16]()
            pixels.reserveCapacity(pixelCount * 4)
            for _ in 0..<pixelCount { pixels.append(contentsOf: halfValues) }
            pixels.withUnsafeBufferPointer { pointer in
                if let base = pointer.baseAddress {
                    pooled.texture.replace(
                        region: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0,
                        withBytes: base,
                        bytesPerRow: width * 4 * MemoryLayout<UInt16>.size
                    )
                }
            }
        }
        // Leak-intentional — the texture lives until the test returns. Its
        // identity is pinned to the pool, but we don't want it recycled
        // while the test is still reading from the caller side.
        return pooled.texture
    }

    private func makeScalarTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        value: Float
    ) throws -> any MTLTexture {
        guard let pooled = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            throw MetalUnavailable(reason: "Could not allocate scalar texture.")
        }
        let pixels = [Float](repeating: value, count: width * height)
        pixels.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                pooled.texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<Float>.size
                )
            }
        }
        return pooled.texture
    }

    private func makeCommandQueue(entry: MetalDeviceCacheEntry) throws -> any MTLCommandQueue {
        guard let queue = entry.device.makeCommandQueue() else {
            throw MetalUnavailable(reason: "Could not create test command queue.")
        }
        queue.label = "CorridorKey Test"
        return queue
    }

    private func commitAndWait(_ commandBuffer: any MTLCommandBuffer) throws {
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()
        if let error = commandBuffer.error { throw error }
    }

    private func readFirstPixel(texture: any MTLTexture) -> SIMD4<Float> {
        let pixelFormat = texture.pixelFormat
        let bytesPerRow: Int
        switch pixelFormat {
        case .rgba32Float:
            bytesPerRow = texture.width * 16
            var pixel = SIMD4<Float>(0, 0, 0, 0)
            texture.getBytes(
                &pixel,
                bytesPerRow: bytesPerRow,
                from: MTLRegionMake2D(0, 0, 1, 1),
                mipmapLevel: 0
            )
            return pixel
        case .rgba16Float:
            bytesPerRow = texture.width * 8
            var halfs = [UInt16](repeating: 0, count: 4)
            halfs.withUnsafeMutableBufferPointer { pointer in
                if let base = pointer.baseAddress {
                    texture.getBytes(
                        base,
                        bytesPerRow: bytesPerRow,
                        from: MTLRegionMake2D(0, 0, 1, 1),
                        mipmapLevel: 0
                    )
                }
            }
            return SIMD4<Float>(
                MatteCodec.halfToFloat(halfs[0]),
                MatteCodec.halfToFloat(halfs[1]),
                MatteCodec.halfToFloat(halfs[2]),
                MatteCodec.halfToFloat(halfs[3])
            )
        default:
            return .zero
        }
    }
}

private extension MTLPixelFormat {
    /// Bytes each pixel occupies for formats this test suite uses. Covers
    /// every format the Corridor Key stages may produce as a despeckle
    /// output.
    var bytesPerPixel: Int {
        switch self {
        case .rgba32Float: return 16
        case .rgba16Float: return 8
        case .r32Float: return 4
        case .r16Float: return 2
        default: return 4
        }
    }
}
