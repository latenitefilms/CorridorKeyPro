//
//  CorridorKeyToolboxPlugIn.swift
//  Corridor Key Toolbox
//
//  Main FxPlug entry point. Final Cut Pro instantiates this class through the
//  `ProPlugPlugInList` declaration in Info.plist and routes every tileable
//  effect call through it. Keeping the class thin and delegating work to
//  dedicated helpers keeps the integration layer easy to maintain.
//

import Foundation
import AppKit
import Metal
import CoreMedia
import simd

@objc(CorridorKeyToolboxPlugIn)
final class CorridorKeyToolboxPlugIn: NSObject, FxTileableEffect, FxAnalyzer {

    // MARK: - Stored properties

    /// The FxPlug API manager the host hands us at construction. All runtime
    /// capabilities – parameter creation, retrieval, timing, colour gamut – are
    /// fetched through this manager on demand.
    let apiManager: any PROAPIAccessing

    /// Per-instance renderer. Lazy so that instantiation in `addParameters` is
    /// cheap and so the first touch of Metal happens in the render thread.
    lazy var renderPipeline: RenderPipeline = RenderPipeline()

    /// Rolling timing of the last rendered frame in milliseconds. Used only
    /// for internal performance tracing.
    let lastFrameMilliseconds = AtomicDouble(0)

    /// Per-instance analysis session. Holds the in-flight matte cache while a
    /// forward-analysis pass is running; dropped back to empty once cleanup
    /// has flushed the final result to the Library. Owned by the plug-in
    /// (rather than a separate registry) so it deallocates automatically when
    /// Final Cut Pro releases the instance — previously, a process-level
    /// registry leaked a full clip's worth of mattes per instance.
    let analysisSession = AnalysisSessionState()

    // MARK: - Init

    @objc required init?(apiManager: any PROAPIAccessing) {
        self.apiManager = apiManager
        super.init()
        PluginLog.notice("CorridorKeyToolboxPlugIn instantiated.")

        // Kick off MLX warm-up eagerly on the system default GPU so the
        // bridge is compiled and resident in memory by the time the user
        // clicks Analyse Clip or starts playback. The shared registry
        // dedupes across multiple plug-in instances — adding the effect
        // to ten clips still loads the model once, not ten times.
        //
        // We warm at the Maximum (2048) rung because that's the default
        // Quality setting. If the user picks a smaller rung, the first
        // analyse pass re-warms for that rung; warming the wrong rung
        // eagerly costs a few hundred MB of memory temporarily but
        // doesn't hurt correctness.
        Self.kickOffDefaultWarmup()
    }

    /// Non-isolated helper that asks `SharedMLXBridgeRegistry` to warm
    /// the default rung on the system default Metal device. Safe to call
    /// from `init`; the registry handles concurrency + deduping.
    private static func kickOffDefaultWarmup() {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        guard let entry = try? MetalDeviceCache.shared.entry(for: device) else { return }
        let defaultRung = QualityMode.maximum2048.resolvedInferenceResolution(forLongEdge: 1920)
        SharedMLXBridgeRegistry.shared.beginWarmup(
            deviceRegistryID: device.registryID,
            rung: defaultRung,
            cacheEntry: entry
        )
    }

    deinit {
        PluginLog.notice("CorridorKeyToolboxPlugIn deallocated.")
    }
}

/// Minimal thread-safe wrapper used for publishing a single Double across
/// threads without reaching for a full lock-based class each time.
final class AtomicDouble: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Double

    init(_ initial: Double) { self.value = initial }

    func set(_ newValue: Double) {
        lock.lock(); value = newValue; lock.unlock()
    }

    func read() -> Double {
        lock.lock(); defer { lock.unlock() }
        return value
    }
}
