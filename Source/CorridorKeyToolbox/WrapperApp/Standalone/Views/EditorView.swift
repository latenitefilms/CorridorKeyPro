//
//  EditorView.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Top-level scene for the standalone editor window. The layout is a
//  vertical stack: preview surface → divider → transport bar. The
//  parameters inspector lives in the trailing-side `.inspector`
//  pane that AppKit-style macOS apps use for property panels.
//
//  We deliberately do not use `NavigationSplitView` here — the editor
//  has no leading sidebar. Final Cut Pro's keyer inspector layout
//  drops the sidebar entirely and dedicates the trailing column to
//  parameter controls; this mirrors that.
//

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct EditorView: View {
    @State private var viewModel: EditorViewModel
    @State private var isImporting = false
    @State private var isExporting = false
    @State private var loadFailureMessage: String?
    @State private var isDropTargeted = false

    /// `EditorViewModel` requires a `StandaloneRenderEngine` constructed
    /// at view-creation time so we can fail fast if Metal isn't
    /// available. The view crashes on init when that's the case;
    /// callers should not present this view on machines without a
    /// working GPU. (Apple Silicon Macs always have one.)
    init() {
        do {
            let engine = try StandaloneRenderEngine()
            self._viewModel = State(wrappedValue: EditorViewModel(renderEngine: engine))
        } catch {
            fatalError("CorridorKey by LateNite could not initialise its render engine: \(error.localizedDescription)")
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            previewArea
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            TransportBar(
                viewModel: viewModel,
                onImport: { isImporting = true },
                onClose: viewModel.closeClip,
                onExport: { isExporting = true }
            )
        }
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text(toolbarTitle)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .padding(.horizontal, 24)
            }
        }
        .navigationTitle(toolbarTitle)
        .inspector(isPresented: .constant(true)) {
            InspectorView(viewModel: viewModel)
                .inspectorColumnWidth(min: 300, ideal: 340, max: 400)
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.movie, .video, .quickTimeMovie, .mpeg4Movie],
            allowsMultipleSelection: false
        ) { result in
            handleImportResult(result)
        }
        .sheet(isPresented: $isExporting) {
            ExportSheet(viewModel: viewModel) {
                isExporting = false
            }
        }
        .onChange(of: viewModel.phase) { _, phase in
            if case .loadFailed(let message) = phase {
                loadFailureMessage = message
            }
        }
        .alert(
            "Couldn't load clip",
            isPresented: Binding(
                get: { loadFailureMessage != nil },
                set: { if !$0 { loadFailureMessage = nil } }
            ),
            actions: {
                Button("OK", role: .cancel) { loadFailureMessage = nil }
            },
            message: {
                Text(loadFailureMessage ?? "Unknown error.")
            }
        )
    }

    @ViewBuilder
    private var previewArea: some View {
        switch viewModel.phase {
        case .noClipLoaded:
            EditorEmptyState(
                isDropTargeted: $isDropTargeted,
                onPickFile: { isImporting = true },
                onDrop: handleDrop(providers:)
            )
        case .loadingClip(let url):
            VStack(spacing: 12) {
                ProgressView()
                Text("Opening \(url.lastPathComponent)…")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black)
        case .ready, .loadFailed:
            ZStack(alignment: .top) {
                Color.black.ignoresSafeArea()
                MetalPreviewView(
                    device: viewModel.renderEngine.device,
                    frame: viewModel.latestPreview,
                    aspectFitSize: viewModel.renderSize,
                    backdrop: viewModel.previewBackdrop
                )
                .contextMenu {
                    PreviewBackdropMenu(viewModel: viewModel)
                }
                OnScreenControlOverlay(
                    viewModel: viewModel,
                    renderSize: viewModel.renderSize
                )
                .allowsHitTesting(viewModel.oscTool != .disabled)
                OSCToolbar(viewModel: viewModel)
                    .padding(.top, 12)
                if viewModel.latestPreview == nil {
                    Text("Rendering preview…")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var toolbarTitle: String {
        switch viewModel.phase {
        case .noClipLoaded: return "CorridorKey by LateNite"
        case .loadingClip: return "Loading…"
        case .ready:
            guard let url = viewModel.clipInfo?.url else { return "CorridorKey by LateNite" }
            return url.deletingPathExtension().lastPathComponent
        case .loadFailed: return "Couldn't load clip"
        }
    }

    // MARK: - File handling

    private func handleImportResult(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            loadClip(at: url)
        case .failure(let error):
            loadFailureMessage = error.localizedDescription
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            DispatchQueue.main.async {
                self.loadClip(at: url)
            }
        }
        return true
    }

    private func loadClip(at url: URL) {
        Task {
            await viewModel.loadClip(at: url)
        }
    }
}

/// Right-click context menu on the preview surface. Lets the user
/// pick the backdrop they want behind the keyed image — defaulting
/// to a transparency-aware checkerboard, with solid colour
/// alternatives for matching specific monitors / colour-grade
/// references.
private struct PreviewBackdropMenu: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        Picker("Player Background", selection: $viewModel.previewBackdrop) {
            ForEach(PreviewBackdrop.allCases) { option in
                Label(option.displayName, systemImage: option.systemImage)
                    .tag(option)
            }
        }
        .pickerStyle(.inline)
    }
}

/// Drag-and-drop / import placeholder shown when the editor has no
/// clip loaded. Mirrors the empty-state pattern used in macOS apps
/// like Photos and Final Cut Pro's import sheet.
private struct EditorEmptyState: View {
    @Binding var isDropTargeted: Bool
    let onPickFile: () -> Void
    let onDrop: ([NSItemProvider]) -> Bool

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 18) {
                Image(systemName: "film.stack")
                    .font(.system(size: 64, weight: .light))
                    .foregroundStyle(.secondary)
                Text("Drop a clip here, or import one to begin.")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                Text("Final Cut Pro–quality keying without leaving this app.\nSupports H.264, HEVC, ProRes, and more.")
                    .multilineTextAlignment(.center)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
                Button("Import Clip…", systemImage: "square.and.arrow.down", action: onPickFile)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .padding(.top, 8)
            }
            .padding()
            .padding(40)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(
                        isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.3),
                        style: StrokeStyle(lineWidth: 1.5, dash: [6, 6])
                    )
                    .padding(28)
            )
        }
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted, perform: onDrop)
    }
}
