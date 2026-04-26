//
//  TransportBar.swift
//  Corridor Key Toolbox — Standalone Editor
//
//  Mac-native scrubber + step buttons that sit beneath the preview
//  surface. The slider drives `EditorViewModel.scrub`, which snaps to
//  the nearest source frame; the step buttons advance / rewind by
//  whole frames.
//

import SwiftUI
import CoreMedia

struct TransportBar: View {
    @Bindable var viewModel: EditorViewModel
    let onImport: () -> Void
    let onClose: () -> Void
    let onExport: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                Button("Import…", systemImage: "square.and.arrow.down", action: onImport)
                    .buttonStyle(.bordered)

                Button("Close Clip", systemImage: "xmark.circle", action: onClose)
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.phase.isReady)

                Spacer()

                Button("Step Back", systemImage: "backward.frame.fill", action: { viewModel.step(byFrames: -1) })
                    .buttonStyle(.borderless)
                    .labelStyle(.iconOnly)
                    .disabled(!viewModel.phase.isReady)
                Button("Step Forward", systemImage: "forward.frame.fill", action: { viewModel.step(byFrames: 1) })
                    .buttonStyle(.borderless)
                    .labelStyle(.iconOnly)
                    .disabled(!viewModel.phase.isReady)

                Spacer()

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

    private var scrubberRow: some View {
        HStack(spacing: 10) {
            Text(timeLabel(for: viewModel.playheadTime))
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)

            Slider(
                value: Binding(
                    get: { normalizedPlayhead },
                    set: { viewModel.scrub(toNormalized: $0) }
                ),
                in: 0...1
            )
            .disabled(!viewModel.phase.isReady)

            Text(timeLabel(for: viewModel.clipInfo?.duration ?? .zero))
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
        }
    }

    private var normalizedPlayhead: Double {
        guard let info = viewModel.clipInfo, info.duration.seconds > 0 else { return 0 }
        return min(max(viewModel.playheadTime.seconds / info.duration.seconds, 0), 1)
    }

    private func timeLabel(for time: CMTime) -> String {
        guard time.isValid && time.isNumeric else { return "00:00:00" }
        let total = max(time.seconds, 0)
        let totalSeconds = Int(total)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60
        let centiseconds = Int((total - Double(totalSeconds)) * 100)
        return String(format: "%02d:%02d:%02d.%02d", hours, minutes, seconds, centiseconds)
    }
}
