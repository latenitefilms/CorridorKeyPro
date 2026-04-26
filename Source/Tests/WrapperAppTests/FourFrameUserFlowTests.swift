//
//  FourFrameUserFlowTests.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  End-to-end coverage that mirrors the user's hand-test against the
//  short ProRes 4444 fixture in `LLM Resources/Benchmark/NikoDruid/
//  Niko - 4 frames.mov`. Each test exercises one slice of the UI
//  flow with the same APIs the SwiftUI views call:
//
//  * `loadClip(at:)` — confirms the user-visible flow works against
//    the same file format the user reported the bug on.
//  * Playback transport — `togglePlayback`, the loop toggle, and the
//    step buttons, including the wrap-around behaviour at the end of
//    a 4-frame clip.
//  * OSC overlay — `handleOSCClick(atNormalizedPoint:)` for foreground,
//    background, and erase tools.
//  * Matte orientation — confirms the cached matte is in
//    AVFoundation top-left convention so it composites correctly with
//    the source frames.
//  * Full round-trip — analyse → export to ProRes 4444 → re-open the
//    output file with `AVURLAsset` to confirm it's playable. This is
//    the gate the user actually cares about.
//

import Testing
import Foundation
import Metal
import CoreMedia
import CoreVideo
import CoreImage
import AVFoundation
@testable import CorridorKeyToolboxApp

@Suite("Four-frame user flow", .serialized)
struct FourFrameUserFlowTests {

    @Test("loads the 4-frame ProRes fixture, renders the first preview frame, and reports the right metadata")
    @MainActor
    func loadsFourFrameFixture() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL(),
            "Niko - 4 frames.mov is missing — see RealClipFixture for the expected path.")
        let viewModel = try await loadFixtureIntoViewModel(at: url)
        #expect(viewModel.totalFrames == 4)
        #expect(viewModel.renderSize.width > 0)
        #expect(viewModel.renderSize.height > 0)
        try #require(viewModel.latestPreview != nil,
                     "First preview frame never landed on the main actor.")
    }

    @Test("step transport advances and clamps at the last frame")
    @MainActor
    func stepTransportClampsAtEnds() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)

        viewModel.step(byFrames: 1)
        #expect(currentFrameIndex(viewModel) == 1)
        viewModel.step(byFrames: 5)
        #expect(currentFrameIndex(viewModel) == viewModel.totalFrames - 1, "Step should clamp at the last frame.")
        viewModel.step(byFrames: -10)
        #expect(currentFrameIndex(viewModel) == 0, "Step should clamp at frame 0.")
    }

    @Test("togglePlayback advances frames and stops at the end when loop is disabled")
    @MainActor
    func togglePlaybackStopsAtEnd() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)
        viewModel.loopEnabled = false

        viewModel.togglePlayback()
        #expect(viewModel.isPlaying == true)

        // Wait long enough for the playback timer to walk through
        // all four frames at the clip's frame rate. NikoDruid's
        // 4-frame ProRes is 24 fps so 4 frames = ~167 ms; pad to
        // 800 ms to absorb the timer's tail.
        let deadline = Date().addingTimeInterval(2.0)
        while viewModel.isPlaying && Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(viewModel.isPlaying == false, "Playback should auto-stop at the last frame when looping is off.")
        #expect(currentFrameIndex(viewModel) == viewModel.totalFrames - 1)
    }

    @Test("toggleLoop wraps the playhead back to zero on the next tick after the end")
    @MainActor
    func togglePlaybackLoopsBackToStart() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)
        viewModel.loopEnabled = true

        // Position one frame before the end so the next playback
        // tick will trigger the loop branch.
        viewModel.step(byFrames: viewModel.totalFrames - 2)
        viewModel.togglePlayback()
        // Wait through one full play loop's worth of ticks.
        let deadline = Date().addingTimeInterval(1.5)
        var sawWrap = false
        while Date() < deadline {
            try await Task.sleep(for: .milliseconds(40))
            if currentFrameIndex(viewModel) == 0 {
                sawWrap = true
                break
            }
        }
        viewModel.togglePlayback()
        #expect(sawWrap, "Loop-enabled playback never wrapped back to frame 0.")
    }

    @Test("OSC click writes a foreground hint at the clicked normalised point")
    @MainActor
    func oscWritesForegroundHint() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)

        viewModel.oscTool = .foregroundHint
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.42, y: 0.61))
        let firstHint = try #require(viewModel.state.hintPointSet.points.first)
        #expect(firstHint.kind == .foreground)
        #expect(abs(firstHint.x - 0.42) < 1.0e-6)
        #expect(abs(firstHint.y - 0.61) < 1.0e-6)
    }

    @Test("OSC erase removes the closest existing hint and disabled tool is a no-op")
    @MainActor
    func oscEraseRemovesNearestHint() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)

        viewModel.oscTool = .foregroundHint
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.5, y: 0.5))
        viewModel.oscTool = .backgroundHint
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.1, y: 0.1))
        try #require(viewModel.state.hintPointSet.points.count == 2)

        viewModel.oscTool = .eraseHint
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.51, y: 0.49))
        #expect(viewModel.state.hintPointSet.points.count == 1)
        #expect(viewModel.state.hintPointSet.points.first?.kind == .background)

        viewModel.oscTool = .disabled
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.0, y: 0.0))
        #expect(viewModel.state.hintPointSet.points.count == 1, "Disabled tool should be a no-op.")
    }

    @Test("clearAllHints empties the hint set in one call")
    @MainActor
    func oscClearAllRemovesEverything() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)
        viewModel.oscTool = .foregroundHint
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.1, y: 0.1))
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.5, y: 0.5))
        viewModel.handleOSCClick(atNormalizedPoint: CGPoint(x: 0.9, y: 0.9))
        try #require(viewModel.state.hintPointSet.points.count == 3)
        viewModel.clearAllHints()
        #expect(viewModel.state.hintPointSet.isEmpty)
    }

    @Test("analysing the 4-frame fixture stores the matte in the same orientation as the source frame's green-bias")
    @MainActor
    func analysingProducesUprightMatte() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)

        let analyseHandle = viewModel
        let collector = MatteCacheCollector()
        let videoSource = try #require(try await getVideoSource(viewModel))
        let runner = AnalysisRunner(renderEngine: viewModel.renderEngine,
                                     videoSource: videoSource)
        await runner.run(state: viewModel.state) { event in
            if case let .frameProcessed(index, _, entry) = event {
                collector.set(index: index, entry: entry)
            }
        }
        let mattes = collector.snapshot()
        try #require(mattes.count == 4)
        let firstMatte = try #require(mattes[0])
        let decoded = try #require(MatteCodec.decode(firstMatte.blob))
        let alphaArray = decoded.alpha
        let width = decoded.width
        let height = decoded.height

        // Probe the source frame at the same resolution and compare
        // the visual "is this background-green" mask to the matte's
        // "is this not-the-key-colour" mask. The two should be
        // strongly anti-correlated in the same orientation:
        // background-green pixels in the source should align with
        // ~0 alpha in the matte, and non-green pixels should align
        // with ~1 alpha. If the matte is y-flipped relative to the
        // source the correlation goes the wrong way and the
        // assertion fires.
        let pixelBuffer = try await videoSource.makeFrame(atTime: .zero)
        let backgroundMask = try Self.greenBackgroundMask(
            from: pixelBuffer,
            sampleWidth: width,
            sampleHeight: height
        )
        precondition(backgroundMask.count == alphaArray.count)
        var alignedAgreement = 0   // matte≈0 where source is green, matte≈1 where source isn't
        var inverseAgreement = 0   // matte≈0 where source is NOT green (would mean matte upside-down)
        for index in alphaArray.indices {
            let matte = alphaArray[index]
            let isBackground = backgroundMask[index]
            if isBackground && matte < 0.4 { alignedAgreement += 1 }
            if !isBackground && matte > 0.6 { alignedAgreement += 1 }
            if isBackground && matte > 0.6 { inverseAgreement += 1 }
            if !isBackground && matte < 0.4 { inverseAgreement += 1 }
        }
        #expect(alignedAgreement > inverseAgreement,
                "Matte aligns with source orientation in only \(alignedAgreement) pixels but disagrees in \(inverseAgreement) — the matte is still inverted relative to the source.")
        _ = analyseHandle
    }

    @Test("full round-trip: load → analyse → export ProRes 4444 → re-open the result", .timeLimit(.minutes(5)))
    @MainActor
    func fullRoundTripExport() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let viewModel = try await loadFixtureIntoViewModel(at: url)

        let collector = MatteCacheCollector()
        let videoSourceForAnalysis = try #require(try await getVideoSource(viewModel))
        let runner = AnalysisRunner(
            renderEngine: viewModel.renderEngine,
            videoSource: videoSourceForAnalysis
        )
        await runner.run(state: viewModel.state) { event in
            if case let .frameProcessed(index, _, entry) = event {
                collector.set(index: index, entry: entry)
            }
        }
        let mattes = collector.snapshot()
        try #require(mattes.count == 4)

        let outputURL = FileManager.default.temporaryDirectory
            .appending(path: "CorridorKeyTests-FourFrame-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let snapshot = ExportProjectSnapshot(state: viewModel.state, cachedMattes: mattes)
        let videoSourceForExport = try #require(try await getVideoSource(viewModel))
        let exporter = ProResExporter(
            renderEngine: viewModel.renderEngine,
            videoSource: videoSourceForExport,
            project: snapshot
        )
        let events = EventCollector<ExportRunnerEvent>()
        await exporter.run(options: ExportOptions(
            destination: outputURL,
            codec: .proRes4444,
            preserveAlpha: true
        )) { event in
            events.append(event)
        }
        let snapshotEvents = events.snapshot()
        let completed = snapshotEvents.contains { event in
            if case .completed = event { return true }
            return false
        }
        #expect(completed, "Export did not complete: \(snapshotEvents)")
        #expect(FileManager.default.fileExists(atPath: outputURL.path))

        // Re-open the exported file: this catches malformed movies
        // that complete the writer but produce a header AV
        // Foundation can't parse on import.
        let writtenAsset = AVURLAsset(url: outputURL)
        let writtenTracks = try await writtenAsset.loadTracks(withMediaType: .video)
        #expect(writtenTracks.count == 1)
        let writtenDuration = try await writtenAsset.load(.duration)
        #expect(writtenDuration.seconds > 0.0)
    }

    // MARK: - Helpers

    @MainActor
    private func loadFixtureIntoViewModel(at url: URL) async throws -> EditorViewModel {
        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: url)
        try #require(viewModel.phase.isReady, "Loading the fixture should leave the editor ready, instead got: \(viewModel.phase)")
        // Wait for the first preview frame so subsequent transport
        // tests have a stable starting state.
        let deadline = Date().addingTimeInterval(20)
        while viewModel.latestPreview == nil && Date() < deadline {
            try await Task.sleep(for: .milliseconds(40))
        }
        return viewModel
    }

    @MainActor
    private func currentFrameIndex(_ viewModel: EditorViewModel) -> Int {
        let fps = max(Double(viewModel.clipInfo?.nominalFrameRate ?? 1), 0.001)
        return Int((viewModel.playheadTime.seconds * fps).rounded())
    }

    /// `EditorViewModel.videoSource` is private; reach it through a
    /// compatible path that tests can use without adding test-only
    /// API to the production type. We can re-create one from the
    /// clip URL because `VideoSource` is cheap to load (header
    /// parsing only — no per-frame decoding yet).
    @MainActor
    private func getVideoSource(_ viewModel: EditorViewModel) async throws -> VideoSource? {
        guard let url = viewModel.clipInfo?.url else { return nil }
        return try await VideoSource(url: url)
    }

    private func meanCoverage(_ alpha: [Float], width: Int, rowsRange: Range<Int>) -> Double {
        var total: Double = 0
        var count = 0
        for row in rowsRange {
            for col in 0..<width {
                total += Double(alpha[row * width + col])
                count += 1
            }
        }
        return count > 0 ? total / Double(count) : 0
    }

    /// Resamples the given pixel buffer to `(sampleWidth, sampleHeight)`
    /// and returns a per-pixel boolean array marking pixels that look
    /// like the green-screen background. Used to anchor the matte
    /// orientation test against the source's known pixel content
    /// instead of guessing where the subject sits in the frame.
    private static func greenBackgroundMask(
        from pixelBuffer: CVPixelBuffer,
        sampleWidth: Int,
        sampleHeight: Int
    ) throws -> [Bool] {
        let context = CIContext(options: [.useSoftwareRenderer: false])
        let extent = CGRect(x: 0, y: 0, width: sampleWidth, height: sampleHeight)
        let scaledImage = CIImage(cvPixelBuffer: pixelBuffer)
            .transformed(by: CGAffineTransform(
                scaleX: CGFloat(sampleWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
                y: CGFloat(sampleHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            ))

        let scratchAttrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        var scratch: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            sampleWidth,
            sampleHeight,
            kCVPixelFormatType_32BGRA,
            scratchAttrs as CFDictionary,
            &scratch
        )
        guard let scratch else { return Array(repeating: false, count: sampleWidth * sampleHeight) }
        context.render(scaledImage, to: scratch, bounds: extent, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))

        CVPixelBufferLockBaseAddress(scratch, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(scratch, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(scratch) else {
            return Array(repeating: false, count: sampleWidth * sampleHeight)
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(scratch)
        var mask = Array(repeating: false, count: sampleWidth * sampleHeight)
        for y in 0..<sampleHeight {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for x in 0..<sampleWidth {
                let pixel = row.advanced(by: x * 4)
                let blue = Float(pixel[0]) / 255
                let green = Float(pixel[1]) / 255
                let red = Float(pixel[2]) / 255
                // Heuristic: green dominates, red+blue are low. Same
                // shape the renderer's `corridorKeyGreenHintKernel`
                // uses to bias its rough matte.
                let isGreen = green > 0.4 && green > red + 0.05 && green > blue + 0.05
                mask[y * sampleWidth + x] = isGreen
            }
        }
        return mask
    }
}
