//
//  CorridorKeyHeaderView.swift
//  CorridorKey by LateNite
//
//  SwiftUI header drawn at the top of the Final Cut Pro inspector. Pulls its
//  data from `CorridorKeyInspectorBridge` so the SwiftUI tree stays free of
//  FxPlug types and can be previewed / unit-tested in isolation.
//

import SwiftUI
import AppKit

/// Inspector header the FxPlug custom-UI parameter hosts. Shows the app icon
/// / version, exposes Analyse / Reset actions, and surfaces the same
/// Analysis fields as the Standalone Editor.
@MainActor
struct CorridorKeyHeaderView: View {

    /// `@State` pins the bridge to this view's identity so SwiftUI can
    /// keep the per-property observation alive across re-renders. With a
    /// plain `let` the subscription was only as stable as the caller's
    /// ownership — and when Final Cut Pro collapsed and re-expanded the
    /// inspector row, the struct was recycled before the hosting view was,
    /// dropping the timer and leaving the header blank. `@State` mirrors
    /// the old `@StateObject` ownership semantics for `@Observable`
    /// classes.
    @State private var bridge: CorridorKeyInspectorBridge

    init(bridge: CorridorKeyInspectorBridge) {
        _bridge = State(wrappedValue: bridge)
    }

    private let applicationIcon: NSImage = CorridorKeyInspectorAssets.applicationIcon()
    private let versionLabel: String = {
        "v\(CorridorKeyInspectorAssets.versionString()) (Build \(CorridorKeyInspectorAssets.buildString()))"
    }()

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(nsImage: applicationIcon)
                .resizable()
                .interpolation(.high)
                .frame(width: 48, height: 48)
                .accessibilityLabel("CorridorKey by LateNite")

            VStack(alignment: .leading, spacing: 6) {
                Text("CorridorKey by LateNite")
                    .font(.headline)
                    .lineLimit(1)
                    .fixedSize(horizontal: false, vertical: true)

                Text(versionLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .fixedSize(horizontal: false, vertical: true)

                analyseControls

                analysisProgressLine

                neuralModelStatusRow

                backendStatusRow

                cachedMattesStatusRow
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 0)
        .task(id: ObjectIdentifier(bridge)) {
            // SwiftUI cancels this task automatically when the view leaves
            // the hierarchy, which removes the Timer-retain-cycle risk the
            // old `startPolling` / `stopPolling` pair had.
            bridge.refreshSnapshot()
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(750))
                if Task.isCancelled { break }
                bridge.refreshSnapshot()
            }
        }
    }

    // MARK: - Subviews

    private var analyseControls: some View {
        HStack(spacing: 8) {
            Button(action: bridge.triggerAnalysis) {
                Label("Analyse Clip", systemImage: "wand.and.stars")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isAnalysisInFlight)

            Button(action: bridge.resetAnalysis) {
                Label("Reset Analysis", systemImage: "trash")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isAnalysisInFlight)

            if case .warming = bridge.snapshot.warmup {
                Button(action: bridge.cancelWarmup) {
                    Label("Cancel", systemImage: "xmark.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    @ViewBuilder
    private var analysisProgressLine: some View {
        switch bridge.snapshot.state {
        case .notAnalysed:
            statusBadge(
                systemImage: "info.circle",
                tint: .secondary,
                text: "Press Analyse Clip to build the matte cache."
            )
        case .requested:
            statusBadge(
                systemImage: "clock.fill",
                tint: .secondary,
                text: "Analysis queued…"
            )
        case .running:
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: bridge.snapshot.progress)
                    .progressViewStyle(.linear)
                statusBadge(
                    systemImage: "wand.and.stars",
                    tint: .secondary,
                    text: runningStatusText
                )
            }
        case .completed:
            statusBadge(
                systemImage: "checkmark.seal.fill",
                tint: .green,
                text: "Analysed \(bridge.snapshot.analyzedFrameCount) \(bridge.snapshot.analyzedFrameCount == 1 ? "frame" : "frames")."
            )
        case .interrupted:
            statusBadge(
                systemImage: "exclamationmark.octagon.fill",
                tint: .orange,
                text: "Analysis interrupted."
            )
        }
    }

    private var neuralModelStatusRow: some View {
        statusBadge(
            systemImage: warmupIconName,
            tint: warmupColor,
            text: warmupLabel
        )
    }

    private var backendStatusRow: some View {
        statusBadge(
            systemImage: "info.circle",
            tint: .secondary,
            text: "Backend: \(bridge.snapshot.renderBackendDescription)"
        )
    }

    private var cachedMattesStatusRow: some View {
        statusBadge(
            systemImage: "square.stack.3d.up",
            tint: .secondary,
            text: "Cached mattes: \(bridge.snapshot.analyzedFrameCount) / \(bridge.snapshot.totalFrameCount)"
        )
    }

    private func statusBadge(
        systemImage: String,
        tint: Color,
        text: String
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Image(systemName: systemImage)
                .foregroundStyle(tint)
            Text(text)
                .foregroundStyle(tint)
        }
        .font(.caption)
        .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Helpers

    private var isAnalysisInFlight: Bool { bridge.snapshot.isWorking }

    private var runningStatusText: String {
        let snapshot = bridge.snapshot
        if snapshot.totalFrameCount > 0 {
            return "Analysing frame \(snapshot.analyzedFrameCount) of \(snapshot.totalFrameCount)…"
        }
        return "Analysing…"
    }

    private var warmupIconName: String {
        switch bridge.snapshot.warmup {
        case .cold: return "moon.zzz"
        case .warming: return "clock"
        case .ready: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    private var warmupColor: Color {
        switch bridge.snapshot.warmup {
        case .cold: return .secondary
        case .warming: return .orange
        case .ready: return .green
        case .failed: return .red
        }
    }

    private var warmupLabel: String {
        switch bridge.snapshot.warmup {
        case .cold: return "Neural Model: Cold"
        case .warming(let resolution): return "Neural Model: Loading (\(resolution)px)…"
        case .ready(let resolution): return "Neural Model: Ready (\(resolution)px)"
        case .failed(let message): return "Neural Model Failed: \(message)"
        }
    }
}
