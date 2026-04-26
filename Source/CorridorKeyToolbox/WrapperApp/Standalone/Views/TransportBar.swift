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

            Text("\(currentFrameLabel) / \(totalFrameLabel)")
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .trailing)
        }
    }

    private var normalizedPlayhead: Double {
        guard let info = viewModel.clipInfo, info.duration.seconds > 0 else { return 0 }
        return min(max(viewModel.playheadTime.seconds / info.duration.seconds, 0), 1)
    }

    private var currentFrameLabel: String {
        guard let info = viewModel.clipInfo else { return "0" }
        let frameRate = max(Double(info.nominalFrameRate), 0.001)
        let frame = Int((viewModel.playheadTime.seconds * frameRate).rounded())
        // Clamp the displayed frame to the clip's total frame count.
        // The asset's duration sits one frame-duration past the
        // last sample, so without the cap the label can briefly
        // read "N+1 / N f" when scrubbing all the way to the right.
        let displayed = max(1, min(frame + 1, viewModel.totalFrames))
        return String(displayed)
    }

    private var totalFrameLabel: String {
        viewModel.totalFrames > 0 ? "\(viewModel.totalFrames) f" : "—"
    }

    /// Wraps the SMPTE timecode helper so the transport bar uses
    /// the same formatter the unit tests cover.
    private func timeLabel(for time: CMTime) -> String {
        let frameRate = Double(viewModel.clipInfo?.nominalFrameRate ?? 24)
        return TransportTimecodeFormatter(frameRate: frameRate).format(time)
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
