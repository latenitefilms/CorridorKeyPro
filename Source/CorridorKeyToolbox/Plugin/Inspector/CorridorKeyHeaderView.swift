//
//  CorridorKeyHeaderView.swift
//  Corridor Key Toolbox
//
//  SwiftUI header drawn at the top of the Final Cut Pro inspector. Pulls its
//  data from `CorridorKeyInspectorBridge` so the SwiftUI tree stays free of
//  FxPlug types and can be previewed / unit-tested in isolation.
//

import SwiftUI
import AppKit

/// Inspector header the FxPlug custom-UI parameter hosts. Shows the app icon
/// / version, exposes Analyse / Reset actions, and surfaces the current
/// analysis status with a live progress line.
@MainActor
struct CorridorKeyHeaderView: View {

    /// `@StateObject` pins the bridge to this view's identity so SwiftUI can
    /// keep the ObservableObject subscription alive across re-renders. With
    /// `@ObservedObject` the subscription was only as stable as the caller's
    /// ownership — and when Final Cut Pro collapsed and re-expanded the
    /// inspector row, the struct was recycled before the hosting view was,
    /// dropping the Published timer and leaving the header blank.
    @StateObject private var bridge: CorridorKeyInspectorBridge

    init(bridge: CorridorKeyInspectorBridge) {
        _bridge = StateObject(wrappedValue: bridge)
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
                .accessibilityLabel("Corridor Key Toolbox")

            VStack(alignment: .leading, spacing: 6) {
                Text("Corridor Key Toolbox")
                    .font(.headline)
                    .lineLimit(1)
                    .fixedSize(horizontal: false, vertical: true)

                Text(versionLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .fixedSize(horizontal: false, vertical: true)

                analyseControls

                statusLine
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
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
                Label("Analyse Clip", systemImage: "waveform.path.ecg.magnifyingglass")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isAnalysisInFlight)

            Button(action: bridge.resetAnalysis) {
                Label("Reset", systemImage: "trash")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isAnalysisInFlight)
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        switch bridge.snapshot.state {
        case .notAnalysed:
            let analysed = bridge.snapshot.analyzedFrameCount
            let total = bridge.snapshot.totalFrameCount
            if total > 0 && analysed >= total {
                statusBadge(
                    systemImage: "checkmark.seal.fill",
                    tint: .green,
                    text: "Cached \(analysed) frames at \(bridge.snapshot.inferenceResolution)px."
                )
            } else if total > 0 {
                statusBadge(
                    systemImage: "exclamationmark.triangle.fill",
                    tint: .orange,
                    text: "Cached \(analysed) of \(total) frames at \(bridge.snapshot.inferenceResolution)px."
                )
            } else {
                statusBadge(
                    systemImage: "exclamationmark.triangle.fill",
                    tint: .orange,
                    text: "Not analysed yet. Click Analyse Clip for real-time playback."
                )
            }
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
                    systemImage: "waveform.path.ecg.magnifyingglass",
                    tint: .secondary,
                    text: runningStatusText
                )
            }
        case .completed:
            statusBadge(
                systemImage: "checkmark.seal.fill",
                tint: .green,
                text: "Analysed \(bridge.snapshot.analyzedFrameCount) frames at \(bridge.snapshot.inferenceResolution)px."
            )
        case .interrupted:
            statusBadge(
                systemImage: "exclamationmark.octagon.fill",
                tint: .orange,
                text: "Analysis interrupted — click Analyse Clip to resume."
            )
        }
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
            return "\(snapshot.analyzedFrameCount)/\(snapshot.totalFrameCount) frames at \(snapshot.inferenceResolution)px…"
        }
        return "Analysing at \(snapshot.inferenceResolution)px…"
    }

}
