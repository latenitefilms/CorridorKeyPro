//
//  TransportPolishTests.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  Targeted regressions for the round of UI polish that landed in
//  v1.0:
//
//  * Frame counter — for an N-frame clip the "current / total"
//    label must never read N+1, even when the playhead lands on
//    the asset's exact duration (which is one frame past the last
//    sample's presentation time).
//  * Timecode formatting — HH:MM:SS:FF (non-drop-frame), not the
//    older HH:MM:SS.ss centisecond format.
//  * Preview-backdrop state — the right-click context menu mutates
//    `EditorViewModel.previewBackdrop`, so the view binding has to
//    be readable and writable from any actor.
//

import Testing
import Foundation
import CoreMedia
@testable import CorridorKeyByLateNiteApp

@Suite("Transport polish", .serialized)
struct TransportPolishTests {

    @Test("nearestFrameTime clamps to the last valid frame so the counter never overshoots")
    func nearestFrameTimeClampsToLastFrame() async throws {
        let synthURL = try await SyntheticVideoFixture.writeMP4(frameCount: 4, fps: 24)
        defer { try? FileManager.default.removeItem(at: synthURL.deletingLastPathComponent()) }
        let source = try await VideoSource(url: synthURL)
        try #require(source.totalFrameCount() == 4)
        let durationSeconds = source.info.duration.seconds
        // The asset's duration is exactly `frameCount / fps` — one
        // frame-duration past the last sample's presentation time.
        // Asking for a snap at duration must still resolve to the
        // last sample (frame index 3), not frame index 4.
        let snappedAtDuration = source.nearestFrameTime(
            to: CMTime(seconds: durationSeconds, preferredTimescale: 24_000)
        )
        let snappedFrameIndex = Int((snappedAtDuration.seconds * 24.0).rounded())
        #expect(snappedFrameIndex == 3, "nearestFrameTime should cap at the last valid frame index, got \(snappedFrameIndex).")

        // Past-the-end inputs collapse onto the last frame too, so
        // a long scrub or a programmatic seek that overshoots
        // never produces an "off-by-one" counter glimpse.
        let snappedPastEnd = source.nearestFrameTime(
            to: CMTime(seconds: durationSeconds * 5, preferredTimescale: 24_000)
        )
        #expect(Int((snappedPastEnd.seconds * 24.0).rounded()) == 3)
    }

    @Test("scrubbing the editor view model to the right end never exceeds the total frame count")
    @MainActor
    func scrubToEndStaysAtLastFrame() async throws {
        let synthURL = try await SyntheticVideoFixture.writeMP4(frameCount: 4, fps: 24)
        defer { try? FileManager.default.removeItem(at: synthURL.deletingLastPathComponent()) }
        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: synthURL)
        try #require(viewModel.phase.isReady)
        try #require(viewModel.totalFrames == 4)

        viewModel.scrub(toNormalized: 1.0)
        let frameIndex = Int((viewModel.playheadTime.seconds * 24).rounded())
        #expect(frameIndex == 3, "Scrub-to-end produced frame index \(frameIndex); the transport bar would read 'N+1 / N f'.")
    }

    /// Slider position 1.0 must put the playhead on the **last
    /// frame**. Repeats the round-trip across every frame index
    /// so short clips can't sneak in a regression at the boundary
    /// — the legacy duration-based mapping topped out at 0.75 on
    /// a 4-frame clip, hiding the final sample behind the right
    /// edge of the slider.
    @Test("Slider reaches the last frame on a 4-frame clip and round-trips at every frame")
    @MainActor
    func sliderRoundTripsToFinalFrame() async throws {
        let synthURL = try await SyntheticVideoFixture.writeMP4(frameCount: 4, fps: 24)
        defer { try? FileManager.default.removeItem(at: synthURL.deletingLastPathComponent()) }
        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        await viewModel.loadClip(at: synthURL)
        try #require(viewModel.phase.isReady)
        try #require(viewModel.totalFrames == 4)

        // Walk every position in [0, 1] in 1/3 steps (the natural
        // granularity of a 4-frame clip) and confirm we land on
        // the right frame.
        for (index, fraction) in [0.0, 1.0/3.0, 2.0/3.0, 1.0].enumerated() {
            viewModel.scrub(toNormalized: fraction)
            let landedFrame = Int((viewModel.playheadTime.seconds * 24).rounded())
            #expect(landedFrame == index,
                    "Slider \(fraction) should snap to frame \(index); landed on \(landedFrame).")
        }
    }

    /// The fps-indicator state on `EditorViewModel` defaults to
    /// zero outside of playback so the transport bar's badge is
    /// safely hidden. `targetPlaybackFPS` only populates once
    /// playback starts — a freshly-launched editor with no clip
    /// must not flash a stray "0 fps" badge.
    @Test("Playback fps state is zero before any clip loads")
    @MainActor
    func playbackFPSStartsZero() async throws {
        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        #expect(viewModel.measuredPlaybackFPS == 0)
        #expect(viewModel.targetPlaybackFPS == 0)
        #expect(viewModel.isPlaying == false)
    }

    @Test("preview backdrop defaults to checkerboard and round-trips through PreviewBackdropMenu cases")
    @MainActor
    func previewBackdropDefaultsToCheckerboardAndRoundTrips() async throws {
        let engine = try StandaloneRenderEngine()
        let viewModel = EditorViewModel(renderEngine: engine)
        #expect(viewModel.previewBackdrop == .checkerboard,
                "Preview backdrop should default to checkerboard so the matte transparency is visible without configuration.")
        for backdrop in PreviewBackdrop.allCases {
            viewModel.previewBackdrop = backdrop
            #expect(viewModel.previewBackdrop == backdrop)
            #expect(!backdrop.displayName.isEmpty)
            #expect(!backdrop.systemImage.isEmpty)
        }
    }

    @Test("PreviewBackdrop covers the requested option set")
    func previewBackdropCoversRequiredOptions() {
        let names = Set(PreviewBackdrop.allCases.map(\.displayName))
        #expect(names.contains("Checkerboard"))
        #expect(names.contains("White"))
        #expect(names.contains("Black"))
        #expect(names.contains("Yellow"))
        #expect(names.contains("Red"))
    }

    @Test("HH:MM:SS:FF timecode formatter")
    func smpteTimecodeFormatter() {
        // Exercise the same helper the transport bar uses to render
        // the time label. Working at 24 fps so frame counts within a
        // second cap at 23 — a non-drop-frame setup matches the
        // timecode editors expect to see in standalone NLE tools.
        let formatter = TransportTimecodeFormatter(frameRate: 24)
        #expect(formatter.format(CMTime(seconds: 0.0, preferredTimescale: 24_000)) == "00:00:00:00")
        #expect(formatter.format(CMTime(seconds: 1.0 / 24, preferredTimescale: 24_000)) == "00:00:00:01")
        #expect(formatter.format(CMTime(seconds: 23.0 / 24, preferredTimescale: 24_000)) == "00:00:00:23")
        #expect(formatter.format(CMTime(seconds: 1.0, preferredTimescale: 24_000)) == "00:00:01:00")
        #expect(formatter.format(CMTime(seconds: 65.5, preferredTimescale: 24_000)) == "00:01:05:12")
        // Negative / invalid time values should clamp to zero so the
        // label never displays nonsense (e.g. negative frame numbers).
        #expect(formatter.format(.invalid) == "00:00:00:00")
        #expect(formatter.format(CMTime(seconds: -1, preferredTimescale: 24_000)) == "00:00:00:00")
    }
}
