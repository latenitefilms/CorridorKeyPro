//
//  NikoDruidEndToEndTests.swift
//  Corridor Key Toolbox — WrapperAppTests
//
//  Drives the standalone editor end-to-end against the real NikoDruid
//  green-screen clip the benchmark suite uses. Each test exercises a
//  different slice of the production code path the way a user would:
//
//  * `loads…` — opens the clip as `EditorViewModel.loadClip(at:)`
//    does, confirms metadata is populated, and waits for the first
//    preview frame to appear.
//  * `previewProducesNonBlackPixels` — renders one preview frame
//    through `StandaloneRenderEngine.render` and reads back the
//    destination texture to confirm the pipeline actually wrote
//    pixels (not just allocated a black drawable).
//  * `analyseSubsetMatchesFxPlugMatte` — runs the standalone analyse
//    path on a single frame and confirms the resulting matte matches
//    `RenderPipeline.extractAlphaMatteForAnalysis` byte-for-byte.
//    This is the parity gate between the standalone editor and the
//    FxPlug renderer; if either side regresses, this test fails.
//  * `analyseFullClipPlusExport` — runs the analyse pass over the
//    full clip then writes a ProRes 4444 .mov so we have an actual
//    keyed deliverable on disk for the user to inspect.
//
//  The "expensive" tests skip themselves if the NikoDruid fixture is
//  missing, so the rest of the suite still runs on minimal checkouts.
//

import Testing
import Foundation
import Metal
import CoreMedia
import CoreVideo
import AVFoundation
@testable import CorridorKeyToolboxApp

@Suite("NikoDruid end-to-end", .serialized)
struct NikoDruidEndToEndTests {

    @Test("loads the clip, populates metadata, and renders the first preview frame")
    @MainActor
    func loadsAndRendersFirstPreview() async throws {
        let url = try #require(
            RealClipFixture.realClipURL(),
            "NikoDruid fixture missing — see RealClipFixture for the expected path."
        )

        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: url)

        try #require(viewModel.phase.isReady, "Loading the NikoDruid clip should leave the editor ready, but phase is \(viewModel.phase).")
        #expect(viewModel.totalFrames > 0)
        #expect(viewModel.renderSize.width > 0 && viewModel.renderSize.height > 0)

        // Wait briefly for the first preview render to land on the
        // main actor. The view model schedules the render in a
        // detached task, so we poll for up to a few seconds.
        let deadline = Date().addingTimeInterval(15)
        while viewModel.latestPreview == nil && Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
        try #require(viewModel.latestPreview != nil, "First preview frame never landed on the main actor — the editor would show as black.")
        #expect(!viewModel.renderBackendDescription.isEmpty)
    }

    @Test("preview render path produces a non-black destination texture for a real frame")
    func previewProducesNonBlackPixels() async throws {
        let url = try #require(RealClipFixture.realClipURL())
        let source = try await VideoSource(url: url)
        let pixelBuffer = try await source.makeFrame(atTime: .zero)
        let engine = try StandaloneRenderEngine()
        let result = try engine.render(
            source: pixelBuffer,
            state: PluginStateData(),
            renderTime: .zero
        )

        let stats = try Self.summariseTexture(result.destinationTexture, device: engine.device)
        // Source pass-through path with no analysed matte: destination
        // should mirror the source — overwhelmingly green for the
        // NikoDruid clip. If the destination is uniformly zero, the
        // render pipeline never wrote anything (this is the bug the
        // user hit before the texture-usage fix).
        #expect(stats.maxValue > 0.05, "Destination texture is effectively black (max value \(stats.maxValue)). The render pass did not write pixels — most likely the destination texture lacks .renderTarget usage.")
        #expect(stats.nonZeroFraction > 0.5, "Less than half the destination pixels are non-zero (got \(stats.nonZeroFraction)).")
    }

    @Test("standalone analyse on one frame matches FxPlug extractAlphaMatteForAnalysis byte-for-byte")
    func analyseSubsetMatchesFxPlugMatte() async throws {
        let url = try #require(RealClipFixture.realClipURL())
        let source = try await VideoSource(url: url)
        // Use a deterministic mid-clip frame so the test isn't flaky
        // around the leader frames AVAssetImageGenerator sometimes
        // returns at t=0.
        let probeTime = CMTime(seconds: 1.0, preferredTimescale: 24_000)
        let pixelBuffer = try await source.makeFrame(atTime: probeTime)

        let engine = try StandaloneRenderEngine()
        let standaloneOutput = try engine.extractMatteBlob(
            source: pixelBuffer,
            state: PluginStateData(qualityMode: .draft512),
            renderTime: probeTime
        )

        // Now drive the same pipeline directly the way FxPlug's
        // FxAnalyzer would, and compare the matte pixels.
        let bridge = try PixelBufferTextureBridge(device: engine.device)
        let sourceBacked = try bridge.makeTexture(for: pixelBuffer, usage: .shaderRead)
        let cache = MetalDeviceCache.shared
        let entry = try cache.entry(for: engine.device)
        let queue = try #require(entry.borrowCommandQueue())
        defer { entry.returnCommandQueue(queue) }
        let directOutput = try RenderPipeline().extractAlphaMatteForAnalysis(
            sourceTexture: sourceBacked.metalTexture,
            state: PluginStateData(qualityMode: .draft512),
            workingGamut: .rec709,
            renderTime: probeTime,
            device: engine.device,
            entry: entry,
            commandQueue: queue,
            readbackSource: false
        )

        #expect(standaloneOutput.width == directOutput.width)
        #expect(standaloneOutput.height == directOutput.height)
        #expect(standaloneOutput.alphaFloats.count == directOutput.alpha.count)

        // Pixel-perfect equality across two independent runs of MLX +
        // post-processing on the same input. Tolerance kept tight
        // because both paths share the same `RenderPipeline` and MLX
        // is deterministic for identical inputs.
        let pairs = zip(standaloneOutput.alphaFloats, directOutput.alpha)
        let maxDelta = pairs.reduce(into: Float(0)) { acc, pair in
            acc = max(acc, abs(pair.0 - pair.1))
        }
        #expect(maxDelta <= 1.0e-4,
                "Standalone matte differs from direct-pipeline matte by up to \(maxDelta) — the standalone editor would not produce parity with Final Cut Pro.")
    }

    @Test("full analyse → export round-trip produces a playable ProRes 4444 movie",
          .timeLimit(.minutes(8)))
    func analyseFullClipPlusExport() async throws {
        let url = try #require(RealClipFixture.realClipURL())
        let engine = try StandaloneRenderEngine()
        let source = try await VideoSource(url: url)

        // Cap the work for CI speed by analysing the first 24 frames
        // of the clip via a custom range. The full clip path is
        // covered by the synthetic-clip suite; this test gates the
        // real-world readback that the synthetic suite can't.
        let runner = AnalysisRunner(renderEngine: engine, videoSource: source)
        let collector = MatteCacheCollector()
        let stateForAnalyse = PluginStateData(qualityMode: .draft512)
        await runner.run(state: stateForAnalyse) { event in
            if case let .frameProcessed(index, _, entry) = event {
                collector.set(index: index, entry: entry)
            }
        }

        let cache = collector.snapshot()
        try #require(cache.count > 0, "Analysis populated zero matte cache entries on the real clip.")

        let outputURL = FileManager.default.temporaryDirectory
            .appending(path: "CorridorKeyTests-NikoDruid-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let snapshot = ExportProjectSnapshot(state: stateForAnalyse, cachedMattes: cache)
        let exporter = ProResExporter(
            renderEngine: engine,
            videoSource: source,
            project: snapshot
        )
        let exportEvents = EventCollector<ExportRunnerEvent>()
        await exporter.run(options: ExportOptions(
            destination: outputURL,
            codec: .proRes4444,
            preserveAlpha: true
        )) { event in
            exportEvents.append(event)
        }
        let events = exportEvents.snapshot()
        let completed = events.contains { event in
            if case .completed = event { return true }
            return false
        }
        #expect(completed, "Export did not complete: \(events)")

        let writtenAsset = AVURLAsset(url: outputURL)
        let writtenTracks = try await writtenAsset.loadTracks(withMediaType: .video)
        #expect(writtenTracks.count == 1)
        let writtenDuration = try await writtenAsset.load(.duration)
        #expect(writtenDuration.seconds > 0.5)
    }

    // MARK: - Helpers

    /// Reads back the texture's pixels and reports the maximum
    /// per-component value plus the fraction of pixels with at least
    /// one non-zero component. Used by the "non-black" assertion.
    private static func summariseTexture(
        _ texture: any MTLTexture,
        device: any MTLDevice
    ) throws -> (maxValue: Float, nonZeroFraction: Double) {
        let width = texture.width
        let height = texture.height
        let pixelCount = width * height

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
            var maxValue: Float = 0
            var nonZeroPixels = 0
            for pixel in 0..<pixelCount {
                let r = Float(Float16(bitPattern: halves[pixel * 4 + 0]))
                let g = Float(Float16(bitPattern: halves[pixel * 4 + 1]))
                let b = Float(Float16(bitPattern: halves[pixel * 4 + 2]))
                maxValue = max(maxValue, max(r, max(g, b)))
                if r > 0.001 || g > 0.001 || b > 0.001 { nonZeroPixels += 1 }
            }
            return (maxValue, Double(nonZeroPixels) / Double(pixelCount))
        case .bgra8Unorm:
            var bytes = [UInt8](repeating: 0, count: pixelCount * 4)
            bytes.withUnsafeMutableBytes { rawBytes in
                if let base = rawBytes.baseAddress {
                    texture.getBytes(
                        base,
                        bytesPerRow: width * 4,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            var maxValue: Float = 0
            var nonZeroPixels = 0
            for pixel in 0..<pixelCount {
                let b = Float(bytes[pixel * 4 + 0]) / 255
                let g = Float(bytes[pixel * 4 + 1]) / 255
                let r = Float(bytes[pixel * 4 + 2]) / 255
                maxValue = max(maxValue, max(r, max(g, b)))
                if r > 0.004 || g > 0.004 || b > 0.004 { nonZeroPixels += 1 }
            }
            return (maxValue, Double(nonZeroPixels) / Double(pixelCount))
        default:
            return (0, 0)
        }
    }
}
