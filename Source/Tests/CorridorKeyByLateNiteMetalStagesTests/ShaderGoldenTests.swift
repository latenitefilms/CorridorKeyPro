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
            screenColor: SIMD3<Float>(0.08, 0.84, 0.08),
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

    @Test("Despill Average on a blue-screen pixel removes blue, not green")
    func despillAverageOnBlueScreen() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // A "spillful" blue pixel — high blue, lower red/green. Mirrors
        // the green-screen test above but with the spill on the blue
        // channel, so we can verify the kernel pivots on the dominant
        // colour in `screenColor` instead of always treating green as
        // the screen.
        let input = SIMD4<Float>(0.30, 0.35, 0.75, 1.0)
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
            screenColor: SIMD3<Float>(0.08, 0.16, 0.84),
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

        // Expected: Average on blue → limit = (R+G)/2, spill = max(0, B-limit),
        // B -= spill, R += spill*0.5, G += spill*0.5.
        let expectedLimit: Float = (input.x + input.y) * 0.5
        let expectedSpill = max(0, input.z - expectedLimit)
        let expectedR = input.x + expectedSpill * 0.5
        let expectedG = input.y + expectedSpill * 0.5
        let expectedB = input.z - expectedSpill

        #expect(abs(output.x - expectedR) < 1e-3, "R mismatch: \(output.x) vs \(expectedR)")
        #expect(abs(output.y - expectedG) < 1e-3, "G mismatch: \(output.y) vs \(expectedG)")
        #expect(abs(output.z - expectedB) < 1e-3, "B mismatch: \(output.z) vs \(expectedB)")
        // Sanity check the green channel is *not* the one being driven down.
        #expect(output.y >= input.y - 1e-3, "Green should not have been reduced on a blue-screen despill.")
    }

    @Test("Despill Ultra on a blue-screen pixel does not push the result toward green")
    func despillUltraOnBlueScreenIsNeutralOrRedLeaning() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Sweep a handful of blue-spilled pixels — neutral shoulders, warm
        // skin tones, and saturated pure blue — through the Ultra method
        // with the canonical blue screen reference, and assert that none
        // of them comes out with green as the dominant channel. The bug
        // hypothesis we're refuting: that removing chroma along the blue
        // axis pushes the residual into yellow strongly enough that
        // green becomes the dominant channel on real footage.
        let blueScreen = SIMD3<Float>(0.08, 0.16, 0.84)
        let cases: [(label: String, input: SIMD4<Float>)] = [
            ("strong blue spill on grey base", SIMD4<Float>(0.40, 0.40, 0.70, 1.0)),
            ("subtle blue spill on warm shoulder", SIMD4<Float>(0.50, 0.40, 0.55, 1.0)),
            ("saturated foreground with mild blue spill", SIMD4<Float>(0.55, 0.45, 0.60, 1.0)),
            ("pure canonical blue (full screen)", SIMD4<Float>(0.08, 0.16, 0.84, 1.0)),
            ("dark blue-tinted hair edge", SIMD4<Float>(0.12, 0.10, 0.30, 1.0))
        ]

        let commandQueue = try makeCommandQueue(entry: entry)

        for (label, input) in cases {
            let sourceTexture = try makeColourTexture(
                entry: entry, width: 2, height: 2, pixelFormat: .rgba32Float, colour: input
            )
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                Issue.record("Could not create command buffer for \(label).")
                return
            }
            let pooled = try RenderStages.despill(
                foreground: sourceTexture,
                strength: 1.0,
                method: .ultra,
                screenColor: blueScreen,
                entry: entry,
                commandBuffer: commandBuffer
            )
            guard let pooled else {
                Issue.record("Ultra despill returned nil for \(label).")
                return
            }
            try commitAndWait(commandBuffer)
            let output = readFirstPixel(texture: pooled.texture)
            pooled.returnManually()

            // The spill (blue channel) must come down (or stay flat for
            // pixels with no positive screen-direction projection — but
            // never go up).
            #expect(
                output.z <= input.z + 1e-3,
                "[\(label)] Blue went up after blue-screen Ultra despill: in=\(input.z) out=\(output.z)"
            )
            // The despill must not turn the pixel green-dominant. The
            // user-visible bug would be "I picked Blue, ran Ultra, and my
            // skin tones look green" — we check the post-despill green
            // channel is no higher than the post-despill red channel by
            // more than a small tolerance for genuinely green-leaning
            // inputs.
            let greenLeanBefore = max(input.y - input.x, 0)
            let greenLeanAfter = max(output.y - output.x, 0)
            #expect(
                greenLeanAfter <= greenLeanBefore + 0.02,
                "[\(label)] Ultra despill amplified the green-over-red lean: before=\(greenLeanBefore), after=\(greenLeanAfter), in=\(input), out=\(output)"
            )
            print("Ultra blue-despill \(label): in=\(input) out=\(output)")
        }
    }

    @Test("Spill alpha attenuation pulls matte to zero on full screen pixels and leaves foreground pixels alone")
    func spillAlphaAttenuationCutsScreenLeavesForeground() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let blueScreen = SIMD3<Float>(0.08, 0.16, 0.84)
        // Three (foreground colour, input matte alpha) pairs that
        // together prove the kernel does what despill needs:
        //   1. A pure-screen pixel with bogus 0.5 alpha → must end up
        //      near zero (the bug being fixed: those pixels currently
        //      produce a coloured halo because alpha stays positive).
        //   2. A foreground skin tone with 1.0 alpha → must pass
        //      through unchanged so we don't punch holes in the
        //      keyed subject.
        //   3. A grey edge pixel with 0.5 alpha → must come down only
        //      slightly because the chroma projects only weakly along
        //      the screen direction.
        let cases: [(label: String, fg: SIMD4<Float>, alphaIn: Float, expectedRange: ClosedRange<Float>)] = [
            ("pure blue screen, alpha=0.5", SIMD4<Float>(0.08, 0.16, 0.84, 1.0), 0.5, 0.0...0.05),
            ("warm skin tone, alpha=1.0", SIMD4<Float>(0.55, 0.45, 0.40, 1.0), 1.0, 0.95...1.0),
            ("neutral grey edge, alpha=0.5", SIMD4<Float>(0.50, 0.50, 0.50, 1.0), 0.5, 0.4...0.5)
        ]

        let commandQueue = try makeCommandQueue(entry: entry)
        for (label, fg, alphaIn, range) in cases {
            let foregroundTexture = try makeColourTexture(
                entry: entry, width: 1, height: 1, pixelFormat: .rgba32Float, colour: fg
            )
            let matteTexture = try makeScalarTexture(entry: entry, width: 1, height: 1, value: alphaIn)

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                Issue.record("Could not create command buffer for \(label).")
                return
            }
            let attenuated = try RenderStages.attenuateMatteBySpill(
                matte: matteTexture,
                foreground: foregroundTexture,
                screenColor: blueScreen,
                strength: 1.0,
                entry: entry,
                commandBuffer: commandBuffer
            )
            guard let attenuated else {
                Issue.record("attenuateMatteBySpill returned nil for \(label).")
                return
            }
            try commitAndWait(commandBuffer)

            // The kernel inherits the matte texture's pixel format
            // (r32Float in this test, r16Float in production). Read
            // back as Float — `makeScalarTexture` only allocates
            // r32Float so the test path is always the simple case.
            var attenuatedAlpha: Float = 0
            attenuated.texture.getBytes(
                &attenuatedAlpha,
                bytesPerRow: MemoryLayout<Float>.size,
                from: MTLRegionMake2D(0, 0, 1, 1),
                mipmapLevel: 0
            )
            attenuated.returnManually()

            #expect(
                range.contains(attenuatedAlpha),
                "[\(label)] attenuated alpha \(attenuatedAlpha) outside expected range \(range)"
            )
            print("Spill alpha attenuation [\(label)] → out=\(attenuatedAlpha)")
        }
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
            screenColor: SIMD3<Float>(0.08, 0.84, 0.08),
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

    // MARK: - Chroma hint

    @Test("Chroma hint kernel keys the screen colour and passes the foreground")
    func chromaHint() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Three colours per screen: the screen colour itself (matte ≈ 0),
        // a warm fleshtone (matte ≈ 1), and the *other* screen colour
        // (matte ≈ 1 — must not be keyed by mistake). The third case is
        // the regression gate: with the legacy green-only kernel a blue
        // pixel would've keyed near 0 because it has high blue and low
        // green, but only because the kernel was hard-coded to look at
        // the green channel — the new screen-aware version must treat
        // it as foreground when the chosen screen is green.
        let commandQueue = try makeCommandQueue(entry: entry)

        func testColour(_ rgb: SIMD4<Float>, screenColor: SIMD3<Float>) throws -> Float {
            let source = try makeColourTexture(
                entry: entry, width: 1, height: 1, pixelFormat: .rgba32Float, colour: rgb
            )
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                Issue.record("Could not create command buffer.")
                return -1
            }
            let pooled = try RenderStages.generateChromaHint(
                source: source,
                screenColor: screenColor,
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

        let greenScreen = SIMD3<Float>(0.08, 0.84, 0.08)
        let blueScreen = SIMD3<Float>(0.08, 0.16, 0.84)

        let greenOnGreen = try testColour(SIMD4<Float>(0.05, 0.85, 0.10, 1.0), screenColor: greenScreen)
        let skinOnGreen = try testColour(SIMD4<Float>(0.60, 0.45, 0.40, 1.0), screenColor: greenScreen)
        let blueOnGreen = try testColour(SIMD4<Float>(0.05, 0.10, 0.85, 1.0), screenColor: greenScreen)
        let blueOnBlue = try testColour(SIMD4<Float>(0.05, 0.10, 0.85, 1.0), screenColor: blueScreen)
        let skinOnBlue = try testColour(SIMD4<Float>(0.60, 0.45, 0.40, 1.0), screenColor: blueScreen)
        let greenOnBlue = try testColour(SIMD4<Float>(0.05, 0.85, 0.10, 1.0), screenColor: blueScreen)

        #expect(greenOnGreen < 0.1, "Green pixel on green screen should matte near 0; got \(greenOnGreen).")
        #expect(skinOnGreen > 0.95, "Skin pixel on green screen should matte near 1; got \(skinOnGreen).")
        #expect(blueOnGreen > 0.95, "Blue pixel on green screen must matte near 1 (foreground); got \(blueOnGreen).")
        #expect(blueOnBlue < 0.1, "Blue pixel on blue screen should matte near 0; got \(blueOnBlue).")
        #expect(skinOnBlue > 0.95, "Skin pixel on blue screen should matte near 1; got \(skinOnBlue).")
        #expect(greenOnBlue > 0.95, "Green pixel on blue screen must matte near 1 (foreground); got \(greenOnBlue).")
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

        // Build a 32×32 matte that covers three cases:
        //   - a 20×20 solid block with alpha 1.0 (large, keep)
        //   - a 5×5 speck of alpha 1.0 far from the block (total 25 pixels,
        //     below the area threshold → must be nuked)
        //   - a single lone sub-binarise-threshold pixel (alpha 0.08 < 0.1
        //     so it has label 0 → must be preserved through the filter,
        //     proving soft hair detail isn't destroyed)
        let width = 32
        let height = 32
        var pixels = [Float](repeating: 0, count: width * height)
        for y in 2..<22 {
            for x in 2..<22 {
                pixels[y * width + x] = 1.0
            }
        }
        for y in 26..<31 {
            for x in 26..<31 {
                pixels[y * width + x] = 1.0
            }
        }
        // Soft, sub-threshold pixel far from any bright region.
        pixels[1 * width + 30] = 0.08

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
            areaThreshold: 100,
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
        // Nuked: the 5×5 speck should be gone (25 pixels < 100 threshold).
        let speckIndex = 28 * width + 28
        #expect(resultPixels[speckIndex] < 0.1, "Speck not removed: \(resultPixels[speckIndex])")
        // Preserved: the lone sub-threshold pixel stays at its original
        // alpha. This is the invariant that stops us destroying soft hair
        // edges during despeckle.
        let softEdgeIndex = 1 * width + 30
        #expect(abs(resultPixels[softEdgeIndex] - 0.08) < 0.01,
                "Soft edge was destroyed: expected 0.08, got \(resultPixels[softEdgeIndex])")

        pooled.returnManually()
        readbackTexture.returnManually()
        matte.returnManually()
    }

    // MARK: - Morphology (threadgroup-cached kernels)

    @Test("Custom morphology dilate (radius 2) covers a 5×5 neighbourhood")
    func morphologyDilateRadius2() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // 100×100 matte with a single 1.0 pixel at (50, 50). After dilate
        // by radius 2 the result should be 1.0 in a 5×5 box around (50,50)
        // and 0 elsewhere. Larger size than the threadgroup width (64) so
        // we exercise multi-threadgroup boundary loads.
        let width = 100
        let height = 100
        let matte = try makeOriginPulseTexture(entry: entry, width: width, height: height)
        let intermediate = try makeScalarTexture(entry: entry, width: width, height: height, value: 0)
        let destination = try makeScalarTexture(entry: entry, width: width, height: height, value: 0)

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        try MatteRefiner.applyMorphology(
            source: matte,
            intermediate: intermediate,
            destination: destination,
            radius: 2, // positive = dilate
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        let pixels = readScalarTexture(destination, width: width, height: height)
        // Inside the 5×5 box around (50, 50): expect 1.0
        for dy in -2...2 {
            for dx in -2...2 {
                let index = (50 + dy) * width + (50 + dx)
                #expect(pixels[index] > 0.99,
                        "Dilate failed at offset (\(dx),\(dy)): \(pixels[index])")
            }
        }
        // Outside the box: expect 0
        let farIndex = 10 * width + 10
        #expect(pixels[farIndex] < 0.01, "Far pixel should still be zero: \(pixels[farIndex])")
    }

    @Test("Custom morphology erode (radius 1) shrinks a solid block")
    func morphologyErodeRadius1() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let width = 80
        let height = 80
        // Solid block from (20,20) to (60,60), zero elsewhere.
        let matte = try makeBlockTexture(
            entry: entry, width: width, height: height,
            blockOriginX: 20, blockOriginY: 20, blockWidth: 40, blockHeight: 40
        )
        let intermediate = try makeScalarTexture(entry: entry, width: width, height: height, value: 0)
        let destination = try makeScalarTexture(entry: entry, width: width, height: height, value: 0)

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        try MatteRefiner.applyMorphology(
            source: matte,
            intermediate: intermediate,
            destination: destination,
            radius: -1, // negative = erode
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        let pixels = readScalarTexture(destination, width: width, height: height)
        // Erosion radius 1: block shrinks to (21,21)..(59,59).
        // Centre of the eroded block should still be 1.0.
        let centreIndex = 40 * width + 40
        #expect(pixels[centreIndex] > 0.99, "Centre eroded out: \(pixels[centreIndex])")
        // Original block edge at (20,40): should now be 0.
        let edgeIndex = 40 * width + 20
        #expect(pixels[edgeIndex] < 0.01, "Original edge wasn't eroded: \(pixels[edgeIndex])")
    }

    @Test("Gaussian blur with σ=1 preserves total mass")
    func gaussianBlurPreservesMass() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Single-pixel impulse → after a small Gaussian, total mass should
        // be preserved (modulo edge clamp leakage at ~0). Use a 64×64
        // canvas with the impulse near the centre so the kernel stays
        // entirely inside.
        let width = 64
        let height = 64
        let matte = try makeOriginPulseTextureAt(
            entry: entry, width: width, height: height, x: 32, y: 32
        )
        let intermediate = try makeScalarTexture(entry: entry, width: width, height: height, value: 0)
        let destination = try makeScalarTexture(entry: entry, width: width, height: height, value: 0)

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        // sigma=1 → kernel radius = 2, falls under MPS breakeven and
        // stays on the custom path.
        try MatteRefiner.applyGaussianBlur(
            source: matte,
            intermediate: intermediate,
            destination: destination,
            radiusPixels: 2,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        let pixels = readScalarTexture(destination, width: width, height: height)
        let totalMass = pixels.reduce(0, +)
        // Expect mass conservation within 1% (Gaussian weights normalise to
        // 1 by construction).
        #expect(abs(totalMass - 1.0) < 0.01, "Mass not preserved: total=\(totalMass)")
        // Centre tap should have the largest weight.
        let centre = pixels[32 * width + 32]
        let neighbour = pixels[32 * width + 33]
        #expect(centre > neighbour, "Centre weight not largest: centre=\(centre), neighbour=\(neighbour)")
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

    /// Builds an `r32Float` texture full of zeros with a single 1.0 pixel
    /// at the centre. Used as the impulse input for morphology + blur
    /// golden tests so the expected output is trivially analytically
    /// derivable.
    private func makeOriginPulseTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int
    ) throws -> any MTLTexture {
        return try makeOriginPulseTextureAt(
            entry: entry,
            width: width,
            height: height,
            x: width / 2,
            y: height / 2
        )
    }

    private func makeOriginPulseTextureAt(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        x: Int,
        y: Int
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not allocate impulse texture.")
        }
        var pixels = [Float](repeating: 0, count: width * height)
        pixels[y * width + x] = 1.0
        pixels.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<Float>.size
                )
            }
        }
        return texture
    }

    /// Builds an `r32Float` texture with a solid 1.0 block in the
    /// specified rectangle and 0.0 elsewhere.
    private func makeBlockTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        blockOriginX: Int,
        blockOriginY: Int,
        blockWidth: Int,
        blockHeight: Int
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not allocate block texture.")
        }
        var pixels = [Float](repeating: 0, count: width * height)
        for y in blockOriginY..<min(blockOriginY + blockHeight, height) {
            for x in blockOriginX..<min(blockOriginX + blockWidth, width) {
                pixels[y * width + x] = 1.0
            }
        }
        pixels.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<Float>.size
                )
            }
        }
        return texture
    }

    /// Reads back an `r32Float` texture into a Swift `[Float]`.
    private func readScalarTexture(
        _ texture: any MTLTexture,
        width: Int,
        height: Int
    ) -> [Float] {
        var pixels = [Float](repeating: 0, count: width * height)
        pixels.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<Float>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        return pixels
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
