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
/// / version, exposes Analyse / Reset actions, and surfaces the current
/// warm-up + analysis status with a live progress line and ETA.
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

                warmupBadge

                statusLine

                hintPointsBadge

                renderStatsFooter
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
    private var warmupBadge: some View {
        switch bridge.snapshot.warmup {
        case .cold, .ready:
            EmptyView()
        case .warming(let resolution):
            let resolutionText = resolution > 0 ? "\(resolution)px" : "neural model"
            statusBadge(
                systemImage: "hourglass",
                tint: .blue,
                text: "Loading \(resolutionText) — first play may stutter (about 2–5 seconds)."
            )
        case .failed(let message):
            statusBadge(
                systemImage: "exclamationmark.triangle.fill",
                tint: .red,
                text: "Neural model unavailable: \(message)"
            )
        }
    }

    @ViewBuilder
    private var renderStatsFooter: some View {
        if let milliseconds = bridge.snapshot.lastRenderMilliseconds, milliseconds > 0 {
            // Format with one decimal place using `Double.formatted(_:)`
            // because plain `String` interpolation doesn't accept the
            // `\(value, format:)` shorthand — that's a `LocalizedStringKey`
            // / `Text` feature.
            let formatted = milliseconds.formatted(.number.precision(.fractionLength(1)))
            statusBadge(
                systemImage: "speedometer",
                tint: .secondary,
                text: "Last frame: \(formatted) ms"
            )
        } else {
            EmptyView()
        }
    }

    @ViewBuilder
    private var hintPointsBadge: some View {
        if bridge.snapshot.hintPointCount > 0 {
            statusBadge(
                systemImage: "circle.dotted",
                tint: .blue,
                text: "\(bridge.snapshot.hintPointCount) subject point\(bridge.snapshot.hintPointCount == 1 ? "" : "s") guiding the keyer."
            )
        } else {
            EmptyView()
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
        let base: String
        if snapshot.totalFrameCount > 0 {
            base = "\(snapshot.analyzedFrameCount)/\(snapshot.totalFrameCount) frames at \(snapshot.inferenceResolution)px"
        } else {
            base = "Analysing at \(snapshot.inferenceResolution)px"
        }
        if let eta = snapshot.analysisETASeconds, eta > 0 {
            return "\(base) — \(formatETA(seconds: eta)) remaining"
        }
        return base + "…"
    }

    /// Formats seconds as a compact "mm:ss" / "h:mm:ss" string for the
    /// inspector ETA badge. Returns `"<1s"` for sub-second estimates so the
    /// badge doesn't dance across "0s" / "1s" every refresh.
    private func formatETA(seconds: Double) -> String {
        if seconds < 1 { return "<1s" }
        let totalSeconds = Int(seconds.rounded())
        let minutes = totalSeconds / 60
        let remainderSeconds = totalSeconds % 60
        if minutes >= 60 {
            let hours = minutes / 60
            let remainderMinutes = minutes % 60
            return "\(hours)h \(remainderMinutes)m"
        }
        if minutes > 0 {
            return "\(minutes)m \(remainderSeconds)s"
        }
        return "\(remainderSeconds)s"
    }

}
