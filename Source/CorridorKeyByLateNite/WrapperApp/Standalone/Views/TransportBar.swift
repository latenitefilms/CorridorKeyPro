//
//  TransportBar.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Mac-native transport bar that sits beneath the preview surface.
//  Modelled after QuickTime Player's controls: a centred play / pause
//  button with frame-step buttons either side, a loop toggle on the
//  right, a current-time / duration label, and a scrubber spanning
//  the bar. The leading column hosts the import / close / export
//  workflow buttons; the trailing column hosts the loop and OSC
//  affordances. The slider drives `EditorViewModel.scrub`, which
//  snaps to the nearest source frame.
//

import SwiftUI
import CoreMedia

struct TransportBar: View {
    @Bindable var viewModel: EditorViewModel
    let onImport: () -> Void
    let onClose: () -> Void
    let onExport: () -> Void
    /// Tells the host editor to surface its `.fileImporter` for the
    /// backdrop image. Threaded in as a closure so `BackdropButton`
    /// can stay independent of `EditorView`'s private @State.
    let onPickBackdropImage: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Button("Import…", systemImage: "square.and.arrow.down", action: onImport)
                    .buttonStyle(.bordered)

                Button("Close Clip", systemImage: "xmark.circle", action: onClose)
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.phase.isReady)

                Spacer()

                playbackCluster

                Spacer()

                Toggle(isOn: $viewModel.loopEnabled) {
                    Label("Loop", systemImage: "repeat")
                }
                .toggleStyle(.button)
                .controlSize(.regular)
                .help("Restart playback when the clip ends.")

                ShowOriginalButton(viewModel: viewModel)
                    .disabled(!viewModel.phase.isReady)

                BackdropButton(
                    viewModel: viewModel,
                    onPickImage: onPickBackdropImage
                )

                SubjectHintsMenu(viewModel: viewModel)
                    .disabled(!viewModel.phase.isReady)

                Button("Export…", systemImage: "square.and.arrow.up", action: onExport)
                    .buttonStyle(.borderedProminent)
                    .disabled(!viewModel.phase.isReady || viewModel.exportStatus.inProgress)
            }

            scrubberRow
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(.regularMaterial)
    }

    private var playbackCluster: some View {
        HStack(spacing: 14) {
            Button("Step Back", systemImage: "backward.frame.fill") {
                viewModel.step(byFrames: -1)
            }
            .buttonStyle(.borderless)
            .labelStyle(.iconOnly)
            .disabled(!viewModel.phase.isReady)
            .controlSize(.large)

            Button(
                viewModel.isPlaying ? "Pause" : "Play",
                systemImage: viewModel.isPlaying ? "pause.fill" : "play.fill"
            ) {
                viewModel.togglePlayback()
            }
            .buttonStyle(.borderless)
            .labelStyle(.iconOnly)
            .disabled(!viewModel.phase.isReady)
            .controlSize(.extraLarge)
            .keyboardShortcut(.space, modifiers: [])
            .font(.title2)

            Button("Step Forward", systemImage: "forward.frame.fill") {
                viewModel.step(byFrames: 1)
            }
            .buttonStyle(.borderless)
            .labelStyle(.iconOnly)
            .disabled(!viewModel.phase.isReady)
            .controlSize(.large)
        }
    }

    private var scrubberRow: some View {
        HStack(spacing: 10) {
            Text(timeLabel(for: viewModel.playheadTime))
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 100, alignment: .leading)

            Slider(
                value: Binding(
                    get: { normalizedPlayhead },
                    set: { viewModel.scrub(toNormalized: $0) }
                ),
                in: 0...1
            )
            .disabled(!viewModel.phase.isReady)

            VStack(alignment: .trailing, spacing: 2) {
                Text(frameCounterLabel)
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
                fpsIndicator
            }
            .frame(width: 110, alignment: .trailing)
        }
    }

    /// Slider position interpreted as a fraction of the clip's
    /// **frame range** (frame 0 → 0, last frame → 1). Earlier
    /// builds mapped onto position-within-duration, which left the
    /// slider one tick short of the final sample on short clips —
    /// a 4-frame clip topped out at 0.75 so the last frame was
    /// visually unreachable.
    private var normalizedPlayhead: Double {
        guard let info = viewModel.clipInfo else { return 0 }
        let lastFrameIndex = max(viewModel.totalFrames - 1, 0)
        guard lastFrameIndex > 0 else { return 0 }
        let frameRate = max(Double(info.nominalFrameRate), 0.001)
        let frameIndex = Int((viewModel.playheadTime.seconds * frameRate).rounded())
        let clamped = max(0, min(frameIndex, lastFrameIndex))
        return Double(clamped) / Double(lastFrameIndex)
    }

    /// Compact "current/total frames" label. Earlier builds rendered
    /// "1 / 4 f" with cramped spacing and a single-letter unit; the
    /// new format reads as a fraction with a spelt-out unit so a
    /// reader who hasn't seen the app before can parse it
    /// immediately ("1/4 frames" vs the previous "1 / 4 f").
    private var frameCounterLabel: String {
        guard viewModel.totalFrames > 0, let info = viewModel.clipInfo else { return "—" }
        let frameRate = max(Double(info.nominalFrameRate), 0.001)
        let frame = Int((viewModel.playheadTime.seconds * frameRate).rounded())
        // Clamp the displayed frame to the clip's total frame count.
        // The asset's duration sits one frame-duration past the
        // last sample, so without the cap the label can briefly
        // read "N+1/N frames" when scrubbing all the way to the right.
        let displayed = max(1, min(frame + 1, viewModel.totalFrames))
        return "\(displayed)/\(viewModel.totalFrames) frames"
    }

    /// Live "fps" badge underneath the frame counter. Always
    /// rendered so the transport bar's vertical extent stays
    /// constant — toggling the row in/out of the layout when
    /// playback starts visibly nudged the slider and surrounding
    /// chrome by a few pixels each tick. The state-to-colour
    /// mapping:
    ///
    /// * paused / stopped → "0fps" in white. Default text colour
    ///   instead of green/orange because zero is not a measurement,
    ///   it's just the resting label.
    /// * playing & ≥ 95 % of target → green. The 5 % slack absorbs
    ///   scheduler jitter so a steady realtime playback doesn't
    ///   flash orange between frames.
    /// * playing & below target → orange. The clip is still showing
    ///   every frame; it's just running slower than realtime.
    private var fpsIndicator: some View {
        let achieved = viewModel.measuredPlaybackFPS
        let target = viewModel.targetPlaybackFPS
        let isPlaying = viewModel.isPlaying
        let isRealtime = isPlaying && target > 0 && achieved >= target * 0.95
        let badgeColour: Color = isPlaying ? (isRealtime ? .green : .orange) : .white
        return Text("\(Int(achieved.rounded()))fps")
            .font(.caption.monospacedDigit())
            .foregroundStyle(badgeColour)
            .help(fpsIndicatorTooltip(isPlaying: isPlaying, isRealtime: isRealtime))
    }

    private func fpsIndicatorTooltip(isPlaying: Bool, isRealtime: Bool) -> String {
        if !isPlaying {
            return "Playback is paused. Press Play to see the live frame rate."
        }
        if isRealtime {
            return "Playing back at the source clip's frame rate."
        }
        return "GPU is slower than the source frame rate — every frame still displays, but slower than realtime."
    }

    /// Wraps the SMPTE timecode helper so the transport bar uses
    /// the same formatter the unit tests cover.
    private func timeLabel(for time: CMTime) -> String {
        let frameRate = Double(viewModel.clipInfo?.nominalFrameRate ?? 24)
        return TransportTimecodeFormatter(frameRate: frameRate).format(time)
    }
}

/// "Show Original" push-and-hold button. While the user holds the
/// pointer down the preview reverts to the unmodified source frame;
/// releasing flips it back to whatever Output mode the inspector is
/// set to. Lets the user A/B the keyed result against the source
/// without losing their parameter pick.
///
/// SwiftUI's `Button` only fires on click-up, so the press / release
/// edges come from a zero-distance `DragGesture` attached
/// `simultaneously`. The gesture's `onChanged` fires once for the
/// initial press (and again for tiny mouse jitter, which is harmless
/// because we set the same flag). `onEnded` fires on release / drag
/// out. Both edges write to the view model, which re-renders the
/// preview.
struct ShowOriginalButton: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        Label("Show Original", systemImage: "eye")
            .labelStyle(.iconOnly)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(viewModel.isShowingOriginal ? Color.accentColor.opacity(0.3) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.secondary.opacity(0.4), lineWidth: 0.5)
            )
            .contentShape(.rect)
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .local)
                    .onChanged { _ in
                        if !viewModel.isShowingOriginal {
                            viewModel.isShowingOriginal = true
                        }
                    }
                    .onEnded { _ in
                        viewModel.isShowingOriginal = false
                    }
            )
            .help("Press and hold to compare with the unkeyed source frame.")
    }
}

/// Formats a `CMTime` as `HH:MM:SS:FF` where `FF` is the frame
/// number within the current second (zero-indexed). Mirrors the
/// non-drop-frame SMPTE timecode editors expect to see in pro
/// video apps. Pulled out into its own struct so unit tests can
/// drive it directly without standing up a transport bar view.
struct TransportTimecodeFormatter {
    let frameRate: Double

    func format(_ time: CMTime) -> String {
        guard time.isValid && time.isNumeric else { return "00:00:00:00" }
        let safeFrameRate = max(frameRate, 0.001)
        let seconds = max(time.seconds, 0)
        // Snap the input time to the nearest whole-frame interval
        // before splitting it apart so the label always matches the
        // frame label on the right of the scrubber, even when the
        // playhead picked up a sub-frame value from a slider drag.
        let totalFrames = Int((seconds * safeFrameRate).rounded())
        let framesPerSecond = max(1, Int(safeFrameRate.rounded()))
        let totalSeconds = totalFrames / framesPerSecond
        let frameWithinSecond = totalFrames % framesPerSecond
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secondsInMinute = totalSeconds % 60
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secondsInMinute, frameWithinSecond)
    }
}
