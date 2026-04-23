//
//  CorridorKeyAnalysisSnapshot.swift
//  Corridor Key Toolbox
//
//  Value-type summary the inspector header renders. Kept dependency-free so
//  it can be unit-tested from the Swift Package side of the repo and stays
//  safe to pass between main-actor SwiftUI views and the FxPlug bridge.
//

import Foundation

struct CorridorKeyAnalysisSnapshot: Equatable, Sendable {
    enum State: Sendable, Equatable {
        case notAnalysed
        case requested
        case running
        case completed
        case interrupted
    }

    let state: State
    let analyzedFrameCount: Int
    let totalFrameCount: Int
    let inferenceResolution: Int
    let warmup: WarmupStatus
    /// Rolling ETA in seconds until analysis completes, or `nil` when we
    /// don't have enough data yet. Populated from the coordinator's rolling
    /// per-frame wall-time tracker.
    let analysisETASeconds: Double?

    /// Default-valued init so the legacy call sites (snapshot logic tests)
    /// keep compiling. `warmup` defaults to `.cold` and `analysisETASeconds`
    /// to `nil`, which matches the pre-v1.0 behaviour before this file
    /// gained these fields.
    init(
        state: State,
        analyzedFrameCount: Int,
        totalFrameCount: Int,
        inferenceResolution: Int,
        warmup: WarmupStatus = .cold,
        analysisETASeconds: Double? = nil
    ) {
        self.state = state
        self.analyzedFrameCount = analyzedFrameCount
        self.totalFrameCount = totalFrameCount
        self.inferenceResolution = inferenceResolution
        self.warmup = warmup
        self.analysisETASeconds = analysisETASeconds
    }

    static let empty = CorridorKeyAnalysisSnapshot(
        state: .notAnalysed,
        analyzedFrameCount: 0,
        totalFrameCount: 0,
        inferenceResolution: 0,
        warmup: .cold,
        analysisETASeconds: nil
    )

    /// Progress fraction in the range `0…1`. Returns `0` when no frames are
    /// expected so the header can fall back to a textual status label instead
    /// of drawing an indeterminate progress bar.
    var progress: Double {
        guard totalFrameCount > 0 else { return 0 }
        return min(1.0, Double(analyzedFrameCount) / Double(totalFrameCount))
    }

    /// `true` when the analyser is actively working, so the header can
    /// disable the Analyse / Reset buttons to avoid double-starts.
    var isWorking: Bool {
        switch state {
        case .requested, .running:
            return true
        case .notAnalysed, .completed, .interrupted:
            return false
        }
    }
}
