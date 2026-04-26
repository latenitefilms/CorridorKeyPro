//
//  VisionHintEngineTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Smoke + behavioural tests for `VisionHintEngine`. Vision is gated on
//  macOS 14+; on older hosts the tests early-return so CI on legacy runners
//  doesn't false-fail. The test fixtures synthesise a tiny "subject vs
//  green-screen" texture, run Vision, and check that the returned mask
//  has the right dimensionality and is non-zero in the subject region.
//

import Testing
import Metal
import simd
import AVFoundation
import CoreImage
import CorridorKeyToolboxLogic
@testable import CorridorKeyToolboxMetalStages

@Suite(.serialized)
struct VisionHintEngineTests {

    /// Smoke test: engine builds, processes a synthetic "subject" texture,
    /// and either returns a mask with the right format or returns nil
    /// (Vision sometimes finds no salient subject in synthetic scenes).
    /// Either outcome is acceptable — we mostly want to confirm that the
    /// engine doesn't crash and that the lifetime of the wrapped texture
    /// is sane.
    @Test func engineProducesMaskOrNilOnSyntheticScene() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let engine = try VisionHintEngine(cacheEntry: entry)

        let width = 256
        let height = 256
        let source = try makeSubjectTexture(entry: entry, width: width, height: height)

        // Vision may return nil for very synthetic content. Both outcomes
        // are valid — what matters is that we don't crash and that any
        // mask we get back is the expected pixel format.
        if let mask = try engine.generateMask(source: source) {
            #expect(mask.texture.pixelFormat == .r8Unorm)
            #expect(mask.texture.width > 0)
            #expect(mask.texture.height > 0)
        }
    }

    /// Confirms that `releaseCachedResources` is a no-op when nothing has
    /// been cached and that calling it after a `generateMask` doesn't
    /// invalidate further `generateMask` calls.
    @Test func releaseCachedResourcesIsIdempotent() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let engine = try VisionHintEngine(cacheEntry: entry)
        engine.releaseCachedResources()
        engine.releaseCachedResources()

        let source = try makeSubjectTexture(entry: entry, width: 64, height: 64)
        _ = try engine.generateMask(source: source)
        engine.releaseCachedResources()
        // Should be able to run again after release — request gets recreated.
        _ = try engine.generateMask(source: source)
    }

    /// Confirms the cache entry's lazy accessor returns the same engine
    /// twice in a row without reinitialising it.
    @Test func cacheEntryReturnsStableEngine() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let first = entry.visionHintEngine() as? VisionHintEngine
        let second = entry.visionHintEngine() as? VisionHintEngine
        #expect(first != nil)
        #expect(first === second)
    }

    /// Real-clip integration test: loads the first frame of the
    /// NikoDruid benchmark clip (a person on a green screen) and
    /// verifies Vision actually detects the subject. This is the
    /// regression gate that catches "Vision returns nothing on real
    /// footage" — exactly the symptom the inspector's Hint
    /// Diagnostic was showing as a black image.
    ///
    /// Skipped (not failed) when the benchmark clip isn't checked
    /// out — the test resource lives in `LLM Resources/Benchmark/`
    /// which is large and not part of the regular SPM bundle.
    @Test func detectsSubjectInNikoDruidFrame() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let engine = try VisionHintEngine(cacheEntry: entry)

        let clipURL = nikoDruidClipURL()
        guard FileManager.default.fileExists(atPath: clipURL.path) else {
            print("Skipping detectsSubjectInNikoDruidFrame: clip not found at \(clipURL.path)")
            return
        }

        let firstFrame = try await firstFrameTexture(
            of: clipURL,
            entry: entry,
            pixelFormat: .rgba16Float
        )

        guard let mask = try engine.generateMask(source: firstFrame) else {
            Issue.record(Comment(rawValue: "Vision returned no foreground instance for the NikoDruid first frame. If this regresses, the Hint Diagnostic will render as black."))
            return
        }

        // Verify the mask isn't all zeros — i.e. the subject was
        // actually detected somewhere in the frame.
        let coverage = try await nonZeroCoverage(of: mask.texture, entry: entry)
        print("NikoDruid Vision mask: \(mask.texture.width)×\(mask.texture.height), non-zero coverage = \(coverage)")
        #expect(coverage > 0.05, "Vision mask should cover at least 5% of the frame for a person on green screen; got \(coverage * 100)%.")
        #expect(coverage < 0.95, "Vision mask should leave at least 5% as background; got \(coverage * 100)%.")
    }

    /// End-to-end diagnostic test: mimics what `renderHintDiagnostic`
    /// does in production — Vision → extractHint → compose with the
    /// `.hint` output mode — and reads back the FINAL composed
    /// destination to confirm it contains red pixels where Vision
    /// detected the subject. This is the test that catches "Hint
    /// Diagnostic renders as black" symptoms.
    @Test(arguments: [
        MTLPixelFormat.rgba16Float,
        MTLPixelFormat.rgba32Float,
        MTLPixelFormat.bgra8Unorm
    ])
    func hintDiagnosticComposeProducesRedOnSubject(
        destinationFormat: MTLPixelFormat
    ) async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let engine = try VisionHintEngine(cacheEntry: entry)

        let clipURL = nikoDruidClipURL()
        guard FileManager.default.fileExists(atPath: clipURL.path) else {
            print("Skipping: clip not found at \(clipURL.path)")
            return
        }
        let firstFrame = try await firstFrameTexture(
            of: clipURL,
            entry: entry,
            pixelFormat: .rgba16Float
        )
        guard let mask = try engine.generateMask(source: firstFrame) else {
            Issue.record("Vision returned nothing.")
            return
        }

        // Step 1 — extractHint to r16Float at source dimensions.
        guard let queue = entry.borrowCommandQueue() else {
            throw MetalUnavailable(reason: "No command queue.")
        }
        defer { entry.returnCommandQueue(queue) }
        guard let extractBuffer = queue.makeCommandBuffer() else {
            throw MetalUnavailable(reason: "No extract command buffer.")
        }
        let hintPooled = try RenderStages.extractHint(
            source: mask.texture,
            layout: 1,
            targetWidth: firstFrame.width,
            targetHeight: firstFrame.height,
            entry: entry,
            commandBuffer: extractBuffer
        )
        try await commit(extractBuffer)

        // Step 2 — compose with `.hint` output mode into a fresh
        // destination texture, exactly mirroring the production
        // diagnostic path.
        let destinationDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: destinationFormat,
            width: firstFrame.width,
            height: firstFrame.height,
            mipmapped: false
        )
        destinationDescriptor.usage = [.renderTarget, .shaderRead]
        destinationDescriptor.storageMode = .shared
        guard let destination = entry.device.makeTexture(descriptor: destinationDescriptor) else {
            throw MetalUnavailable(reason: "No destination texture.")
        }
        guard let composeBuffer = queue.makeCommandBuffer() else {
            throw MetalUnavailable(reason: "No compose command buffer.")
        }
        try inlineComposeWithHintMode(
            matte: hintPooled.texture,
            destination: destination,
            entry: entry,
            commandBuffer: composeBuffer
        )
        try await commit(composeBuffer)
        hintPooled.returnManually()

        // Step 3 — read back the composed destination and verify the
        // red channel has substantial coverage. .hint mode writes
        // (alpha, 0, 0, 1) so the green and blue channels MUST stay
        // zero, while red MUST be non-zero somewhere.
        let (redCoverage, maxRed, maxGreen, maxBlue) = try await measureRGBA(of: destination, entry: entry)
        print("Hint Diagnostic compose [\(destinationFormat.rawValue)]: red coverage=\(redCoverage), maxRed=\(maxRed), maxGreen=\(maxGreen), maxBlue=\(maxBlue)")
        #expect(redCoverage > 0.05, "Compose with .hint mode should produce red on the subject region (>5%); got \(redCoverage * 100)%.")
        #expect(maxRed > 0.5, "At least one pixel should be near-saturated red; got max red \(maxRed).")
        #expect(maxGreen < 0.05, ".hint mode must not write green; got max green \(maxGreen).")
        #expect(maxBlue < 0.05, ".hint mode must not write blue; got max blue \(maxBlue).")
    }

    /// Inline replica of `RenderPipeline.composeInto` using the
    /// `.hint` output mode. Saves us from importing the FxPlug-
    /// dependent RenderPipeline into the SPM test target.
    private func inlineComposeWithHintMode(
        matte: any MTLTexture,
        destination: any MTLTexture,
        entry: MetalDeviceCacheEntry,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        let renderPipelines = try entry.renderPipelines(for: destination.pixelFormat)
        let passDescriptor = MTLRenderPassDescriptor()
        passDescriptor.colorAttachments[0].texture = destination
        passDescriptor.colorAttachments[0].loadAction = .clear
        passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        passDescriptor.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw MetalUnavailable(reason: "No render encoder")
        }
        encoder.label = "Test Compose Hint"
        let tileWidth = Float(destination.width)
        let tileHeight = Float(destination.height)
        let halfW = tileWidth * 0.5
        let halfH = tileHeight * 0.5
        var vertices: [CKVertex2D] = [
            CKVertex2D(position: SIMD2<Float>(halfW, -halfH), textureCoordinate: SIMD2<Float>(1, 1)),
            CKVertex2D(position: SIMD2<Float>(-halfW, -halfH), textureCoordinate: SIMD2<Float>(0, 1)),
            CKVertex2D(position: SIMD2<Float>(halfW, halfH), textureCoordinate: SIMD2<Float>(1, 0)),
            CKVertex2D(position: SIMD2<Float>(-halfW, halfH), textureCoordinate: SIMD2<Float>(0, 0))
        ]
        var viewportSize = SIMD2<UInt32>(UInt32(tileWidth), UInt32(tileHeight))
        encoder.setViewport(MTLViewport(
            originX: 0, originY: 0,
            width: Double(tileWidth), height: Double(tileHeight),
            znear: -1, zfar: 1
        ))
        encoder.setRenderPipelineState(renderPipelines.compose)
        encoder.setVertexBytes(
            &vertices,
            length: MemoryLayout<CKVertex2D>.stride * vertices.count,
            index: Int(CKVertexInputIndexVertices.rawValue)
        )
        encoder.setVertexBytes(
            &viewportSize,
            length: MemoryLayout<SIMD2<UInt32>>.size,
            index: Int(CKVertexInputIndexViewportSize.rawValue)
        )
        encoder.setFragmentTexture(matte, index: Int(CKTextureIndexSource.rawValue))
        encoder.setFragmentTexture(matte, index: Int(CKTextureIndexForeground.rawValue))
        encoder.setFragmentTexture(matte, index: Int(CKTextureIndexMatte.rawValue))
        var params = CKComposeParams(outputMode: OutputMode.hint.shaderValue)
        encoder.setFragmentBytes(
            &params,
            length: MemoryLayout<CKComposeParams>.size,
            index: Int(CKBufferIndexComposeParams.rawValue)
        )
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
    }

    /// Wrap an MTLCommandBuffer commit in async/await.
    private func commit(_ buffer: any MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            buffer.addCompletedHandler { commandBuffer in
                if let error = commandBuffer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            buffer.commit()
        }
    }

    /// Reads back the destination and reports
    /// (coverage of red>0.03, max red, max green, max blue).
    /// Handles rgba16Float, rgba32Float, and bgra8Unorm — the three
    /// destination formats FCP gives us in production via
    /// `metalPixelFormat(for:)`. The texture must be `.shared`
    /// storage (caller's responsibility).
    private func measureRGBA(
        of texture: any MTLTexture,
        entry: MetalDeviceCacheEntry
    ) async throws -> (Double, Float, Float, Float) {
        let width = texture.width
        let height = texture.height
        let pixelCount = width * height

        var redNonZero = 0
        var maxRed: Float = 0
        var maxGreen: Float = 0
        var maxBlue: Float = 0

        switch texture.pixelFormat {
        case .rgba16Float:
            var halves = [UInt16](repeating: 0, count: pixelCount * 4)
            halves.withUnsafeMutableBytes { bytes in
                if let base = bytes.baseAddress {
                    texture.getBytes(
                        base,
                        bytesPerRow: width * 4 * MemoryLayout<UInt16>.size,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            for i in 0..<pixelCount {
                let r = Float(Float16(bitPattern: halves[i * 4 + 0]))
                let g = Float(Float16(bitPattern: halves[i * 4 + 1]))
                let b = Float(Float16(bitPattern: halves[i * 4 + 2]))
                if r > 0.03 { redNonZero += 1 }
                if r > maxRed { maxRed = r }
                if g > maxGreen { maxGreen = g }
                if b > maxBlue { maxBlue = b }
            }
        case .rgba32Float:
            var floats = [Float](repeating: 0, count: pixelCount * 4)
            floats.withUnsafeMutableBytes { bytes in
                if let base = bytes.baseAddress {
                    texture.getBytes(
                        base,
                        bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            for i in 0..<pixelCount {
                let r = floats[i * 4 + 0]
                let g = floats[i * 4 + 1]
                let b = floats[i * 4 + 2]
                if r > 0.03 { redNonZero += 1 }
                if r > maxRed { maxRed = r }
                if g > maxGreen { maxGreen = g }
                if b > maxBlue { maxBlue = b }
            }
        case .bgra8Unorm:
            var bytes8 = [UInt8](repeating: 0, count: pixelCount * 4)
            bytes8.withUnsafeMutableBytes { bytes in
                if let base = bytes.baseAddress {
                    texture.getBytes(
                        base,
                        bytesPerRow: width * 4,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            for i in 0..<pixelCount {
                // BGRA byte order in storage; sample order is R,G,B = [2],[1],[0].
                let b = Float(bytes8[i * 4 + 0]) / 255
                let g = Float(bytes8[i * 4 + 1]) / 255
                let r = Float(bytes8[i * 4 + 2]) / 255
                if r > 0.03 { redNonZero += 1 }
                if r > maxRed { maxRed = r }
                if g > maxGreen { maxGreen = g }
                if b > maxBlue { maxBlue = b }
            }
        default:
            throw MetalUnavailable(reason: "Unhandled pixel format \(texture.pixelFormat.rawValue)")
        }

        return (Double(redNonZero) / Double(pixelCount), maxRed, maxGreen, maxBlue)
    }

    /// Standalone repro: confirms `extractHint(layout: 1)` works on a
    /// synthetic 3840×2160 r8Unorm texture. Lets us isolate "does
    /// extractHint work with a Vision-shaped input?" from "does the
    /// Vision-derived texture work?"
    @Test func extractHintWorksOnSynthetic1KR8Texture() async throws {
        let entry = try TestHarness.makeEntryOrSkip()
        let width = 1024
        let height = 1024
        try await runSyntheticExtractTest(entry: entry, width: width, height: height)
    }

    @Test func extractHintWorksOnSynthetic2KR8Texture() async throws {
        let entry = try TestHarness.makeEntryOrSkip()
        let width = 1920
        let height = 1080
        try await runSyntheticExtractTest(entry: entry, width: width, height: height)
    }

    @Test func extractHintWorksOnSynthetic4KR8Texture() async throws {
        let entry = try TestHarness.makeEntryOrSkip()
        let width = 3840
        let height = 2160
        try await runSyntheticExtractTest(entry: entry, width: width, height: height)
    }

    private func runSyntheticExtractTest(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int
    ) async throws {
        let synthetic = try makeSynthetic4KMaskTexture(entry: entry, width: width, height: height)

        guard let commandQueue = entry.borrowCommandQueue() else {
            throw MetalUnavailable(reason: "No command queue available.")
        }
        defer { entry.returnCommandQueue(commandQueue) }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalUnavailable(reason: "Could not make command buffer.")
        }

        let extracted = try RenderStages.extractHint(
            source: synthetic,
            layout: 1,
            targetWidth: width,
            targetHeight: height,
            entry: entry,
            commandBuffer: commandBuffer
        )

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            commandBuffer.commit()
        }

        let coverage = try await nonZeroCoverageR16Float(of: extracted.texture, entry: entry)
        print("Synthetic-source extracted coverage = \(coverage)")
        #expect(coverage > 0.4 && coverage < 0.6, "Expected ~50% coverage from a half-on/half-off source.")
        extracted.returnManually()
    }

    /// Same clip, but exercises the resample-into-`.r16Float` path
    /// the renderer uses (`extractHint(layout: 1)`) so the
    /// integration covers the full Vision → ExtractHint → r16Float
    /// chain that ends up bound to the compose shader.
    @Test func extractHintFromNikoDruidVisionMaskIsNonZero() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let engine = try VisionHintEngine(cacheEntry: entry)

        let clipURL = nikoDruidClipURL()
        guard FileManager.default.fileExists(atPath: clipURL.path) else {
            print("Skipping extractHintFromNikoDruidVisionMaskIsNonZero: clip not found at \(clipURL.path)")
            return
        }

        let firstFrame = try await firstFrameTexture(
            of: clipURL,
            entry: entry,
            pixelFormat: .rgba16Float
        )
        print("[test] first frame texture: \(firstFrame.width)×\(firstFrame.height) format=\(firstFrame.pixelFormat.rawValue)")
        guard let mask = try engine.generateMask(source: firstFrame) else {
            Issue.record("Vision returned nothing for the NikoDruid first frame.")
            return
        }
        print("[test] vision mask texture: \(mask.texture.width)×\(mask.texture.height) format=\(mask.texture.pixelFormat.rawValue) usage=\(mask.texture.usage.rawValue)")

        guard let commandQueue = entry.borrowCommandQueue() else {
            throw MetalUnavailable(reason: "No command queue available.")
        }
        defer { entry.returnCommandQueue(commandQueue) }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalUnavailable(reason: "Could not make command buffer.")
        }
        print("[test] about to call extractHint")

        // Mirror what `renderHintDiagnostic` does in production:
        // resample the Vision mask through `extractHint(layout: 1)`
        // into an r16Float pooled texture sized to the source frame.
        let extracted = try RenderStages.extractHint(
            source: mask.texture,
            layout: 1,
            targetWidth: firstFrame.width,
            targetHeight: firstFrame.height,
            entry: entry,
            commandBuffer: commandBuffer
        )
        print("[test] extractHint encoded; about to commit")
        mask.retainOnCompletion(of: commandBuffer)

        // Add the completion handler BEFORE committing — Metal asserts
        // when handlers are added after commit. This is also the
        // pattern the production pipeline uses for cooperative async
        // wait without `waitUntilCompleted` (which is unavailable
        // from async test contexts).
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            commandBuffer.commit()
        }

        let coverage = try await nonZeroCoverageR16Float(of: extracted.texture, entry: entry)
        print("Extracted r16Float hint: \(extracted.texture.width)×\(extracted.texture.height), non-zero coverage = \(coverage)")
        #expect(coverage > 0.05, "Resampled hint should preserve the subject region; got \(coverage * 100)%.")
        extracted.returnManually()
    }

    // MARK: - Real-clip helpers

    /// Path to the NikoDruid benchmark clip. The file lives outside
    /// the SPM bundle (it's 130 MB), so we resolve it relative to
    /// this test file's location at compile time using `#filePath`.
    private func nikoDruidClipURL(file: String = #filePath) -> URL {
        // The test file is at:
        //   <repo>/Source/Tests/CorridorKeyToolboxMetalStagesTests/VisionHintEngineTests.swift
        // The clip is at:
        //   <repo>/LLM Resources/Benchmark/NikoDruid/Input.MP4
        // Walk up three levels (.../Tests -> .../Source -> .../<repo>)
        // and tack on the benchmark path.
        let testURL = URL(fileURLWithPath: file)
        let repoRoot = testURL
            .deletingLastPathComponent() // VisionHintEngineTests.swift
            .deletingLastPathComponent() // CorridorKeyToolboxMetalStagesTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // Source
        return repoRoot
            .appendingPathComponent("LLM Resources")
            .appendingPathComponent("Benchmark")
            .appendingPathComponent("NikoDruid")
            .appendingPathComponent("Input.MP4")
    }

    /// Extracts the first frame from a video file as a Metal texture
    /// in the requested pixel format. Uses `AVAssetImageGenerator`
    /// for the frame extraction and a `CIContext` to render into a
    /// freshly-allocated `MTLTexture`.
    private func firstFrameTexture(
        of url: URL,
        entry: MetalDeviceCacheEntry,
        pixelFormat: MTLPixelFormat
    ) async throws -> any MTLTexture {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let cgImage: CGImage = try await withCheckedThrowingContinuation { continuation in
            generator.generateCGImagesAsynchronously(
                forTimes: [NSValue(time: CMTime(value: 0, timescale: 600))]
            ) { _, image, _, result, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard result == .succeeded, let image else {
                    continuation.resume(throwing: MetalUnavailable(reason: "AVAssetImageGenerator did not return a frame"))
                    return
                }
                continuation.resume(returning: image)
            }
        }

        let width = cgImage.width
        let height = cgImage.height
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        descriptor.storageMode = .shared
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not allocate destination texture for first frame.")
        }

        let ciImage = CIImage(cgImage: cgImage)
        let ciContext = CIContext(mtlDevice: entry.device)
        ciContext.render(
            ciImage,
            to: texture,
            commandBuffer: nil,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        return texture
    }

    /// Reads back an r8Unorm texture and reports the fraction of
    /// pixels above a small threshold. Used to verify the Vision
    /// mask is non-empty. Blits to a `.shared` staging texture
    /// because the production Vision mask is `.private` storage.
    private func nonZeroCoverage(
        of texture: any MTLTexture,
        entry: MetalDeviceCacheEntry
    ) async throws -> Double {
        let width = texture.width
        let height = texture.height
        let pixelCount = width * height

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let staging = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not create staging texture for readback")
        }

        guard let queue = entry.borrowCommandQueue() else {
            throw MetalUnavailable(reason: "No command queue for readback")
        }
        defer { entry.returnCommandQueue(queue) }
        guard let buffer = queue.makeCommandBuffer(),
              let blit = buffer.makeBlitCommandEncoder() else {
            throw MetalUnavailable(reason: "Could not create blit encoder for readback")
        }
        blit.copy(from: texture, to: staging)
        blit.endEncoding()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            buffer.addCompletedHandler { commandBuffer in
                if let error = commandBuffer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            buffer.commit()
        }

        var pixels = [UInt8](repeating: 0, count: pixelCount)
        pixels.withUnsafeMutableBytes { bytes in
            if let base = bytes.baseAddress {
                staging.getBytes(
                    base,
                    bytesPerRow: width,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        let nonZero = pixels.reduce(0) { $0 + ($1 > 8 ? 1 : 0) }
        return Double(nonZero) / Double(pixelCount)
    }

    /// Reads back an r16Float texture and reports the fraction of
    /// pixels whose value is above ~3% (matching what the compose
    /// shader's saturate would consider visible).
    /// Blits the texture to a `.shared` staging buffer first because
    /// the texture pool defaults to `.private` storage and `getBytes`
    /// on `.private` textures is forbidden by Metal's validation.
    private func nonZeroCoverageR16Float(
        of texture: any MTLTexture,
        entry: MetalDeviceCacheEntry
    ) async throws -> Double {
        let width = texture.width
        let height = texture.height
        let pixelCount = width * height

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: texture.pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let staging = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not create staging texture for readback")
        }

        guard let queue = entry.borrowCommandQueue() else {
            throw MetalUnavailable(reason: "No command queue for readback")
        }
        defer { entry.returnCommandQueue(queue) }
        guard let buffer = queue.makeCommandBuffer(),
              let blit = buffer.makeBlitCommandEncoder() else {
            throw MetalUnavailable(reason: "Could not create blit encoder for readback")
        }
        blit.copy(from: texture, to: staging)
        blit.endEncoding()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            buffer.addCompletedHandler { commandBuffer in
                if let error = commandBuffer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
            buffer.commit()
        }

        var halfStorage = [UInt16](repeating: 0, count: pixelCount)
        halfStorage.withUnsafeMutableBytes { bytes in
            if let base = bytes.baseAddress {
                staging.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<UInt16>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        var nonZero = 0
        for half in halfStorage {
            let value = Float(Float16(bitPattern: half))
            if value > 0.03 { nonZero += 1 }
        }
        return Double(nonZero) / Double(pixelCount)
    }

    private func makeSynthetic4KMaskTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not allocate synthetic mask texture.")
        }
        var pixels = [UInt8](repeating: 0, count: width * height)
        // Half-on: rows in the bottom half are 255 (subject), top half 0.
        for y in (height / 2)..<height {
            for x in 0..<width {
                pixels[y * width + x] = 255
            }
        }
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: width
            )
        }
        return texture
    }

    // MARK: - Helpers

    /// Builds a tiny RGBA texture with a green-screen background and a
    /// dark "subject" rectangle in the centre. Vision's foreground
    /// detector treats the central rectangle as a salient subject.
    private func makeSubjectTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not create test texture.")
        }

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let index = (y * width + x) * 4
                let isSubject =
                    x > width / 4 && x < width * 3 / 4 &&
                    y > height / 4 && y < height * 3 / 4
                if isSubject {
                    // Skin-toned subject
                    pixels[index] = 200
                    pixels[index + 1] = 160
                    pixels[index + 2] = 120
                    pixels[index + 3] = 255
                } else {
                    // Bright green screen
                    pixels[index] = 30
                    pixels[index + 1] = 200
                    pixels[index + 2] = 30
                    pixels[index + 3] = 255
                }
            }
        }
        let bytesPerRow = width * 4
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: bytesPerRow
            )
        }
        return texture
    }
}
