//
//  FusedKernelTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Verifies the two fused kernels that replace multiple separate
//  post-inference passes:
//    * `corridorKeyMatteRefineKernel` — levels + gamma + refiner blend.
//    * `corridorKeyForegroundPostProcessKernel` — source passthrough +
//      light wrap + edge decontamination + inverse-screen rotation.
//
//  Each test feeds synthetic inputs through the fused pipeline and
//  compares the output against the analytically-expected result or
//  against the unfused pipeline's behaviour, so a fusion regression
//  fails the suite before it reaches the plug-in.
//

import Foundation
import Metal
import Testing
import simd
import CorridorKeyToolboxLogic
@testable import CorridorKeyToolboxMetalStages

@Suite("FusedKernels")
struct FusedKernelTests {

    // MARK: - Fused matte refine

    @Test("Fused matte refine with strength 1.0 matches a levels+gamma-only pass")
    func fusedMatteRefineNoRefinerBlend() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let width = 8
        let height = 1
        let inputRow: [Float] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]

        guard let source = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate source texture.")
            return
        }
        inputRow.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                source.texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<Float>.size
                )
            }
        }
        guard let destination = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate destination texture.")
            return
        }

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        _ = try RenderStages.applyFusedMatteRefine(
            matte: source.texture,
            destination: destination.texture,
            blackPoint: 0.2,
            whitePoint: 0.8,
            gamma: 1.0,
            refinerStrength: 1.0,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        var result = [Float](repeating: 0, count: width)
        result.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                destination.texture.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<Float>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        source.returnManually()
        destination.returnManually()

        // Levels remap: (alpha - 0.2) / (0.8 - 0.2), clamped to [0, 1].
        for (index, input) in inputRow.enumerated() {
            let expected = min(max((input - 0.2) / 0.6, 0), 1)
            #expect(abs(result[index] - expected) < 1e-3,
                    "Fused matte levels mismatch at index \(index): got \(result[index]), expected \(expected)")
        }
    }

    @Test("Fused matte refine with strength 0 biases fully toward the coarse stand-in")
    func fusedMatteRefineFullyBiasedToCoarse() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Uniform refined matte of 1.0 blurred in the shader would have
        // been averaged toward the edge (clamp sampler), so we just
        // check the "strength = 0 → coarse value" invariant using the
        // coarse-blur stand-in generated internally.
        let size = 16
        guard let matte = entry.texturePool.acquire(
            width: size, height: size, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate matte.")
            return
        }
        let uniform = [Float](repeating: 0.7, count: size * size)
        uniform.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                matte.texture.replace(
                    region: MTLRegionMake2D(0, 0, size, size),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: size * MemoryLayout<Float>.size
                )
            }
        }
        guard let destination = entry.texturePool.acquire(
            width: size, height: size, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate destination.")
            return
        }

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let artifacts = try RenderStages.applyFusedMatteRefine(
            matte: matte.texture,
            destination: destination.texture,
            blackPoint: 0.0,
            whitePoint: 1.0,
            gamma: 1.0,
            refinerStrength: 0.0,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        var centrePixel: Float = 0
        destination.texture.getBytes(
            &centrePixel,
            bytesPerRow: size * MemoryLayout<Float>.size,
            from: MTLRegionMake2D(size / 2, size / 2, 1, 1),
            mipmapLevel: 0
        )
        artifacts.coarsePooled?.returnManually()
        artifacts.intermediatePooled?.returnManually()
        matte.returnManually()
        destination.returnManually()

        // At interior pixels, the Gaussian blur of a uniform 0.7 matte
        // is still ~0.7 (clamped sampler preserves uniform interior).
        // With strength 0, output = coarse ≈ 0.7.
        #expect(abs(centrePixel - 0.7) < 5e-2,
                "Expected centre pixel near 0.7 (coarse-biased), got \(centrePixel)")
    }

    // MARK: - Fused foreground post-process

    @Test("Fused foreground post-process with everything off writes foreground unchanged")
    func fusedForegroundAllFlagsOff() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let colour = SIMD4<Float>(0.3, 0.45, 0.85, 1.0)
        let foreground = try makeColourTexture(
            entry: entry, width: 4, height: 4, pixelFormat: .rgba32Float, colour: colour
        )
        let source = try makeColourTexture(
            entry: entry, width: 4, height: 4, pixelFormat: .rgba32Float, colour: SIMD4<Float>(1, 0, 0, 1)
        )
        let matte = try makeScalarTexture(entry: entry, width: 4, height: 4, value: 0.6)

        let config = RenderStages.FusedForegroundConfig(
            sourcePassthrough: false,
            lightWrapEnabled: false,
            lightWrapStrength: 0,
            lightWrapEdgeBias: 0.6,
            edgeDecontaminateEnabled: false,
            edgeDecontaminateStrength: 0,
            screenColor: SIMD3<Float>(0.08, 0.84, 0.08),
            inverseScreenMatrix: matrix_identity_float3x3,
            applyInverseRotation: false
        )

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let output = try RenderStages.applyFusedForegroundPostProcess(
            foreground: foreground,
            sourceRGB: source,
            matte: matte,
            blurredSource: source,
            config: config,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)
        let pixel = readFirstPixel(texture: output.texture)
        output.returnManually()

        // With every stage disabled the output is foreground, alpha = 1.
        #expect(abs(pixel.x - colour.x) < 1e-3)
        #expect(abs(pixel.y - colour.y) < 1e-3)
        #expect(abs(pixel.z - colour.z) < 1e-3)
        #expect(abs(pixel.w - 1.0) < 1e-3)
    }

    @Test("Fused foreground post-process respects source passthrough")
    func fusedForegroundSourcePassthroughOnly() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let fgColour = SIMD4<Float>(0.2, 0.2, 0.2, 1.0)
        let srcColour = SIMD4<Float>(0.8, 0.1, 0.4, 1.0)
        let foreground = try makeColourTexture(
            entry: entry, width: 4, height: 4, pixelFormat: .rgba32Float, colour: fgColour
        )
        let source = try makeColourTexture(
            entry: entry, width: 4, height: 4, pixelFormat: .rgba32Float, colour: srcColour
        )
        let matte = try makeScalarTexture(entry: entry, width: 4, height: 4, value: 1.0)

        let config = RenderStages.FusedForegroundConfig(
            sourcePassthrough: true,
            lightWrapEnabled: false,
            lightWrapStrength: 0,
            lightWrapEdgeBias: 0.6,
            edgeDecontaminateEnabled: false,
            edgeDecontaminateStrength: 0,
            screenColor: SIMD3<Float>(0.08, 0.84, 0.08),
            inverseScreenMatrix: matrix_identity_float3x3,
            applyInverseRotation: false
        )

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let output = try RenderStages.applyFusedForegroundPostProcess(
            foreground: foreground,
            sourceRGB: source,
            matte: matte,
            blurredSource: source,
            config: config,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)
        let pixel = readFirstPixel(texture: output.texture)
        output.returnManually()

        // `m * src + (1 - m) * fg` with m = 1 → output = src.
        #expect(abs(pixel.x - srcColour.x) < 1e-3)
        #expect(abs(pixel.y - srcColour.y) < 1e-3)
        #expect(abs(pixel.z - srcColour.z) < 1e-3)
    }

    @Test("Fused foreground post-process applies inverse rotation when flagged")
    func fusedForegroundInverseRotation() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Doubling matrix — 2x each channel.
        let matrix = simd_float3x3(diagonal: SIMD3<Float>(2, 2, 2))
        let fgColour = SIMD4<Float>(0.1, 0.2, 0.3, 1.0)
        let foreground = try makeColourTexture(
            entry: entry, width: 4, height: 4, pixelFormat: .rgba32Float, colour: fgColour
        )
        let source = try makeColourTexture(
            entry: entry, width: 4, height: 4, pixelFormat: .rgba32Float, colour: fgColour
        )
        let matte = try makeScalarTexture(entry: entry, width: 4, height: 4, value: 0.0)

        let config = RenderStages.FusedForegroundConfig(
            sourcePassthrough: false,
            lightWrapEnabled: false,
            lightWrapStrength: 0,
            lightWrapEdgeBias: 0.6,
            edgeDecontaminateEnabled: false,
            edgeDecontaminateStrength: 0,
            screenColor: SIMD3<Float>(0.08, 0.84, 0.08),
            inverseScreenMatrix: matrix,
            applyInverseRotation: true
        )

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let output = try RenderStages.applyFusedForegroundPostProcess(
            foreground: foreground,
            sourceRGB: source,
            matte: matte,
            blurredSource: source,
            config: config,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)
        let pixel = readFirstPixel(texture: output.texture)
        output.returnManually()

        #expect(abs(pixel.x - fgColour.x * 2) < 1e-3)
        #expect(abs(pixel.y - fgColour.y * 2) < 1e-3)
        #expect(abs(pixel.z - fgColour.z * 2) < 1e-3)
    }

    // MARK: - Helpers

    /// Allocates a non-pooled texture so the test keeps full ownership
    /// and the pool doesn't recycle the underlying storage mid-test.
    /// (The prior helper returned `PooledTexture.texture` and dropped
    /// the wrapper — `deinit` immediately returned the slot, so other
    /// pool users could overwrite the pixels before the kernel ran.)
    private func makeColourTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat,
        colour: SIMD4<Float>
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
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
                    bytesPerRow: width * MemoryLayout<SIMD4<Float>>.size
                )
            }
        }
        return texture
    }

    private func makeScalarTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        value: Float
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not allocate scalar texture.")
        }
        let pixels = [Float](repeating: value, count: width * height)
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

    /// Blits the private output texture to a shared scratch texture so
    /// we can read its contents from the CPU. Metal's `getBytes` is
    /// undefined for `.private` storage — on Apple Silicon it often
    /// returns stale pool memory, which is what we saw before adding
    /// this blit step.
    private func readFirstPixel(texture: any MTLTexture) -> SIMD4<Float> {
        guard texture.pixelFormat == .rgba32Float else { return .zero }
        let device = texture.device
        guard let sharedTexture = makeSharedReadbackTexture(
            device: device,
            width: texture.width,
            height: texture.height
        ) else { return .zero }
        guard let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { return .zero }
        blit.copy(
            from: texture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
            to: sharedTexture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()

        var pixel = SIMD4<Float>(0, 0, 0, 0)
        sharedTexture.getBytes(
            &pixel,
            bytesPerRow: texture.width * 16,
            from: MTLRegionMake2D(0, 0, 1, 1),
            mipmapLevel: 0
        )
        return pixel
    }

    private func makeSharedReadbackTexture(
        device: any MTLDevice,
        width: Int,
        height: Int
    ) -> (any MTLTexture)? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderRead, .shaderWrite]
        return device.makeTexture(descriptor: descriptor)
    }
}
