//
//  WarmupStatus.swift
//  CorridorKey by LateNite
//
//  Lifecycle of the MLX bridge as seen by the UI. Drives the "Loading neural
//  model…" badge in the inspector so users understand the first-play stall
//  without thinking the plug-in is broken.
//

import Foundation

/// Current state of the MLX bridge warm-up. Surfaced into the analysis
/// snapshot the inspector bridge publishes to SwiftUI, so the UI can
/// explain the stall before the first frame keys correctly.
///
/// `public` so the SPM `CorridorKeyToolboxMetalStages` module can return
/// values of this type from `InferenceCoordinator.warmupStatus` and
/// `SharedMLXBridgeRegistry.status(...)`. The Xcode FxPlug target — where
/// everything is in a single module — is unaffected by the visibility
/// promotion.
public enum WarmupStatus: Sendable, Equatable {
    /// MLX has not been touched yet this session.
    case cold
    /// MLX is currently loading/compiling. `resolution` is the requested
    /// rung; `0` means the resolution hasn't been fixed yet.
    case warming(resolution: Int)
    /// MLX is loaded and can serve inference for the given resolution.
    case ready(resolution: Int)
    /// Warm-up failed. `message` is the localized error from MLX, safe to
    /// display in the inspector.
    case failed(String)
}
