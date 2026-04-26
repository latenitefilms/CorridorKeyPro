//
//  ExportSheet.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Modal sheet that gathers ProRes 4444 export options, runs the
//  export through `EditorViewModel.runExport`, and surfaces progress
//  + result so the user knows when their file is ready.
//

import SwiftUI
import AppKit

struct ExportSheet: View {
    @Bindable var viewModel: EditorViewModel
    let dismiss: () -> Void

    @State private var codec: ExportOptions.ProResCodec = .proRes4444
    @State private var preserveAlpha: Bool = true
    @State private var destinationURL: URL?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Export Keyed Clip")
                .font(.title2)
                .bold()

            Text("Renders the current parameters across every analysed frame and writes the result as Apple ProRes inside a QuickTime movie.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 10) {
                EnumPicker(
                    title: "Codec",
                    selection: $codec,
                    onChange: {}
                )

                Toggle("Preserve Alpha Channel", isOn: $preserveAlpha)
                    .disabled(!codec.supportsAlpha)
            }

            HStack(spacing: 8) {
                Button("Choose Destination…", systemImage: "folder", action: chooseDestination)
                    .buttonStyle(.bordered)
                Spacer()
                Text(destinationLabel)
                    .font(.callout)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(.secondary)
            }

            Divider()

            ExportProgressLabel(status: viewModel.exportStatus, totalFrames: viewModel.totalFrames)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) {
                    if viewModel.exportStatus.inProgress {
                        viewModel.cancelExport()
                    }
                    dismiss()
                }
                Button("Export", systemImage: "square.and.arrow.up", action: startExport)
                    .buttonStyle(.borderedProminent)
                    .disabled(destinationURL == nil || viewModel.exportStatus.inProgress)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 460)
    }

    private var destinationLabel: String {
        guard let destinationURL else { return "No file chosen yet" }
        return destinationURL.path
    }

    private func chooseDestination() {
        let panel = NSSavePanel()
        panel.title = "Export As"
        panel.prompt = "Export"
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [.quickTimeMovie]
        panel.nameFieldStringValue = (viewModel.clipInfo?.url.deletingPathExtension().lastPathComponent ?? "Untitled") + " (Keyed).mov"
        let response = panel.runModal()
        guard response == .OK, let url = panel.url else { return }
        destinationURL = url
    }

    private func startExport() {
        guard let destinationURL else { return }
        let options = ExportOptions(
            destination: destinationURL,
            codec: codec,
            preserveAlpha: preserveAlpha
        )
        viewModel.runExport(options: options)
    }
}

extension ExportOptions.ProResCodec: DisplayNamed {}

private struct ExportProgressLabel: View {
    let status: EditorExportStatus
    let totalFrames: Int

    var body: some View {
        switch status {
        case .idle:
            Text("Press Export to start writing the file.")
                .font(.callout)
                .foregroundStyle(.secondary)
        case .running(let processed, let total):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(processed), total: Double(max(total, 1)))
                Text("Encoded \(processed) of \(total) frames")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .completed(let url):
            VStack(alignment: .leading, spacing: 6) {
                Label("Export complete: \(url.lastPathComponent)", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
                .buttonStyle(.borderless)
            }
        case .failed(let message):
            Label("Export failed: \(message)", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
                .fixedSize(horizontal: false, vertical: true)
        case .cancelled:
            Label("Export cancelled.", systemImage: "stop.circle.fill")
                .foregroundStyle(.orange)
        }
    }
}
