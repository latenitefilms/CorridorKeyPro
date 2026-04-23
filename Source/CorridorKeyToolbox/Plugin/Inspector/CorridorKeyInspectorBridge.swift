//
//  CorridorKeyInspectorBridge.swift
//  Corridor Key Toolbox
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
        let (analysed, total, resolution) = currentAnalysisCounts()
        return CorridorKeyAnalysisSnapshot(
            state: state,
            analyzedFrameCount: analysed,
            totalFrameCount: total,
            inferenceResolution: resolution
        )
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

    private func currentAnalysisCounts() -> (analysed: Int, total: Int, resolution: Int) {
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
