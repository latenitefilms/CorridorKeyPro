//
//  CorridorKeyInspectorBridge.swift
//  CorridorKey by LateNite
//
//  Thin bridge between the SwiftUI inspector header and the FxPlug host
//  APIs. Keeps SwiftUI views free of `PROAPIAccessing` / `NSError` plumbing
//  and gives us one place to funnel analyse-state polling through.
//

import Foundation
import AppKit
import CoreMedia

/// Main-actor-isolated bridge owned by the SwiftUI header. We hold the
/// `apiManager` weakly-ish through a retained protocol existential so the
/// host can still tear the plug-in down even while the view is alive.
@MainActor
final class CorridorKeyInspectorBridge: ObservableObject {
    /// Latest analysis snapshot the header draws from.
    @Published private(set) var snapshot: CorridorKeyAnalysisSnapshot = .empty

    private let apiManager: any PROAPIAccessing
    private weak var plugin: CorridorKeyToolboxPlugIn?

    /// Rolling EMA of per-frame analysis wall-time (seconds). Populated by
    /// observing the frame count delta between refreshes; `nil` until we
    /// have at least two samples under the same session.
    private var lastAnalyzedCount: Int = 0
    private var lastRefreshDate: Date?
    private var smoothedPerFrameSeconds: Double?

    init(apiManager: any PROAPIAccessing, plugin: CorridorKeyToolboxPlugIn) {
        self.apiManager = apiManager
        self.plugin = plugin
        refreshSnapshot()
    }

    // MARK: - User actions

    func triggerAnalysis() {
        plugin?.startForwardAnalysisPass()
        refreshSnapshot()
    }

    func resetAnalysis() {
        plugin?.clearAnalysisCache()
        refreshSnapshot()
    }

    /// Cancels in-flight MLX warm-up and resets the ETA estimator. The
    /// warm-up Task is the primary cancelable work — a full forward-
    /// analysis cancel is a host-level action (kFxAnalysisState_*).
    func cancelWarmup() {
        plugin?.renderPipeline.inferenceCoordinator.cancelWarmup()
        refreshSnapshot()
    }

    // MARK: - Snapshot polling

    /// Reads the analysis state from FxPlug and updates `snapshot`. Safe to
    /// call on a timer — it just consults the host and inspects the cached
    /// analysis dictionary, neither of which is expensive. The reads are
    /// wrapped in an `FxCustomParameterActionAPI_v4` bracket because FxPlug
    /// refuses to vend the analysis / parameter APIs to a custom view when
    /// no action is in flight.
    func refreshSnapshot() {
        let updated = readSnapshotWithinActionScope() ?? .empty
        if snapshot != updated {
            snapshot = updated
        }
    }

    // MARK: - Private helpers

    private func readSnapshotWithinActionScope() -> CorridorKeyAnalysisSnapshot? {
        guard let actionAPI = apiManager.api(for: (any FxCustomParameterActionAPI_v4).self) as? any FxCustomParameterActionAPI_v4 else {
            return nil
        }
        actionAPI.startAction(self)
        defer { actionAPI.endAction(self) }

        let state = currentAnalysisState()
        let (analysed, total, resolution) = currentAnalysisCounts(for: state)
        let warmup = plugin?.renderPipeline.inferenceCoordinator.warmupStatus ?? .cold
        let eta = updatedETA(forState: state, analysed: analysed, total: total)
        let lastRender = plugin?.lastFrameMilliseconds.read()
        let lastRenderForFooter: Double? = (lastRender ?? 0) > 0 ? lastRender : nil
        let hintCount = currentHintPointCount()
        return CorridorKeyAnalysisSnapshot(
            state: state,
            analyzedFrameCount: analysed,
            totalFrameCount: total,
            inferenceResolution: resolution,
            warmup: warmup,
            analysisETASeconds: eta,
            lastRenderMilliseconds: lastRenderForFooter,
            hintPointCount: hintCount
        )
    }

    private func currentHintPointCount() -> Int {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            return 0
        }
        var raw: (any NSCopying & NSObjectProtocol & NSSecureCoding)?
        retrieval.getCustomParameterValue(
            &raw,
            fromParameter: ParameterIdentifier.subjectPoints,
            at: CMTime.zero
        )
        return HintPointSet.fromParameterDictionary(raw as? NSDictionary).points.count
    }

    /// Updates the rolling per-frame timer and returns the ETA seconds for
    /// the current snapshot. Resets when analysis isn't running.
    private func updatedETA(forState state: CorridorKeyAnalysisSnapshot.State, analysed: Int, total: Int) -> Double? {
        guard state == .running, analysed < total else {
            lastAnalyzedCount = 0
            lastRefreshDate = nil
            smoothedPerFrameSeconds = nil
            return nil
        }
        let now = Date()
        defer {
            lastAnalyzedCount = analysed
            lastRefreshDate = now
        }
        guard let previousDate = lastRefreshDate else {
            return nil
        }
        let deltaFrames = analysed - lastAnalyzedCount
        let deltaSeconds = now.timeIntervalSince(previousDate)
        guard deltaFrames > 0, deltaSeconds > 0 else {
            // Hold the current estimate if we have one.
            if let smoothed = smoothedPerFrameSeconds {
                let remaining = max(total - analysed, 0)
                return smoothed * Double(remaining)
            }
            return nil
        }
        let sample = deltaSeconds / Double(deltaFrames)
        // Simple exponential moving average — responsive without being
        // twitchy as frames complete at varying latencies.
        let alpha = 0.3
        if let smoothed = smoothedPerFrameSeconds {
            smoothedPerFrameSeconds = smoothed * (1 - alpha) + sample * alpha
        } else {
            smoothedPerFrameSeconds = sample
        }
        let remaining = max(total - analysed, 0)
        return (smoothedPerFrameSeconds ?? sample) * Double(remaining)
    }

    private func currentAnalysisState() -> CorridorKeyAnalysisSnapshot.State {
        guard let analysisAPI = apiManager.api(for: (any FxAnalysisAPI_v2).self) as? any FxAnalysisAPI_v2 else {
            return .notAnalysed
        }
        switch analysisAPI.analysisStateForEffect() {
        case kFxAnalysisState_AnalysisRequested:
            return .requested
        case kFxAnalysisState_AnalysisStarted:
            return .running
        case kFxAnalysisState_AnalysisCompleted:
            return .completed
        case kFxAnalysisState_AnalysisInterrupted:
            return .interrupted
        default:
            return .notAnalysed
        }
    }

    private func currentAnalysisCounts(for state: CorridorKeyAnalysisSnapshot.State) -> (analysed: Int, total: Int, resolution: Int) {
        // While an analysis pass is in flight, the persisted `AnalysisData`
        // inside the custom parameter lags behind: the analyser only flushes
        // it every 10 frames. Read the in-memory session counters instead so
        // the progress bar and resolution label update continuously from
        // frame one.
        if (state == .running || state == .requested),
           let plugin, let live = plugin.liveAnalysisProgress() {
            return live
        }

        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            return (0, 0, 0)
        }
        guard let plugin, let data = plugin.loadAnalysisData(using: retrieval) else {
            return (0, 0, 0)
        }
        // If the user flipped the Quality popup since the cache was built, the
        // stored matte resolution no longer matches the rung they want. Treat
        // that like "nothing analysed yet" so the orange header badge prompts
        // a fresh analyse — the underlying cache is overwritten the moment
        // they click Analyse Clip again.
        var currentQualityRaw: Int32 = Int32(QualityMode.automatic.rawValue)
        retrieval.getIntValue(
            &currentQualityRaw,
            fromParameter: ParameterIdentifier.qualityMode,
            at: CMTime.zero
        )
        if Int(currentQualityRaw) != data.qualityModeRaw {
            return (0, 0, 0)
        }
        return (data.analyzedCount, data.frameCount, data.inferenceResolution)
    }
}

/// Returns the wrapper app icon bundled alongside the FxPlug service. Walks
/// up from the main bundle to find the owning `.app`, matching the lookup
/// Final Cut Pro uses so we get the same artwork the Finder would show.
@MainActor
enum CorridorKeyInspectorAssets {
    static func containingApplicationURL() -> URL? {
        var current = Bundle.main.bundleURL
        while current.path != "/" {
            if current.pathExtension == "app" {
                return current
            }
            current = current.deletingLastPathComponent()
        }
        return nil
    }

    static func containingApplicationBundle() -> Bundle? {
        guard let url = containingApplicationURL() else { return nil }
        return Bundle(url: url)
    }

    static func applicationIcon() -> NSImage {
        if let url = containingApplicationURL() {
            return NSWorkspace.shared.icon(forFile: url.path)
        }
        return NSWorkspace.shared.icon(forFile: Bundle.main.bundlePath)
    }

    static func versionString() -> String {
        let bundle = containingApplicationBundle() ?? .main
        return (bundle.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "1.0"
    }

    static func buildString() -> String {
        let bundle = containingApplicationBundle() ?? .main
        return (bundle.infoDictionary?["CFBundleVersion"] as? String) ?? "1"
    }
}
