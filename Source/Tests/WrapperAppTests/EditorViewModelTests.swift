//
//  EditorViewModelTests.swift
//  Corridor Key Toolbox — WrapperAppTests
//
//  Drives the @Observable view model that owns the editor window.
//  These tests would have caught the original "permission denied"
//  regression — `loadClip(at:)` is exactly the entry point the
//  `.fileImporter` callback in `EditorView` calls, so anything that
//  blocks at the AVURLAsset layer surfaces here.
//

import Testing
import Foundation
import CoreMedia
@testable import CorridorKeyToolboxApp

@Suite("EditorViewModel", .serialized)
@MainActor
struct EditorViewModelTests {

    @Test("loads a synthetic MP4 and transitions to .ready with clip metadata populated")
    func loadsClipFromFileImporterURL() async throws {
        let url = try await SyntheticVideoFixture.writeMP4()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: url)

        #expect(viewModel.phase.isReady, "View model should be ready, instead got: \(viewModel.phase)")
        #expect(viewModel.clipInfo?.url == url)
        #expect(viewModel.totalFrames > 0)
        #expect(viewModel.renderSize == CGSize(width: 320, height: 180))
    }

    @Test("step button advances and rewinds the playhead by whole frames")
    func stepAdvancesPlayheadByWholeFrames() async throws {
        let url = try await SyntheticVideoFixture.writeMP4()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: url)
        let initialFrame = framesAtPlayhead(viewModel)

        viewModel.step(byFrames: 3)
        #expect(framesAtPlayhead(viewModel) == initialFrame + 3)

        viewModel.step(byFrames: -2)
        #expect(framesAtPlayhead(viewModel) == initialFrame + 1)

        // Advancing past the end should clamp to the last frame.
        viewModel.step(byFrames: 999)
        #expect(framesAtPlayhead(viewModel) == viewModel.totalFrames - 1)
    }

    @Test("closeClip returns the editor to the empty state and drops the cache")
    func closeClipResetsState() async throws {
        let url = try await SyntheticVideoFixture.writeMP4()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: url)
        viewModel.matteCache[0] = MatteCacheEntry(
            blob: Data([0]),
            width: 1,
            height: 1,
            inferenceResolution: 512
        )
        viewModel.closeClip()
        #expect(viewModel.phase == .noClipLoaded)
        #expect(viewModel.clipInfo == nil)
        #expect(viewModel.matteCache.isEmpty)
    }

    @Test("loading a clip that doesn't exist surfaces a load failure phase")
    func loadFailureIsRecoverable() async throws {
        let bogus = FileManager.default.temporaryDirectory
            .appending(path: "Definitely-Not-Real-\(UUID().uuidString).mp4")
        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: bogus)
        if case .loadFailed = viewModel.phase {
            // expected
        } else {
            Issue.record("Expected .loadFailed phase, got \(viewModel.phase)")
        }
    }

    @Test("editing a parameter does not crash even with no clip loaded")
    func parameterChangeIsSafeWithoutClip() throws {
        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        viewModel.state.alphaGamma = 1.5
        viewModel.parameterDidChange()
        #expect(viewModel.phase == .noClipLoaded)
    }

    @Test("Quality-Mode change picks an inference resolution within the supported ladder")
    func qualityModeRespectsLadder() {
        for mode in QualityMode.allCases {
            let resolution = mode.resolvedInferenceResolution(forLongEdge: 1920)
            let supported = QualityMode.inferenceLadder
            #expect(supported.contains(resolution),
                    "QualityMode \(mode) resolved to \(resolution), outside the ladder \(supported)")
        }
    }

    private func framesAtPlayhead(_ viewModel: EditorViewModel) -> Int {
        let fps = max(Double(viewModel.clipInfo?.nominalFrameRate ?? 1), 0.001)
        return Int((viewModel.playheadTime.seconds * fps).rounded())
    }
}
