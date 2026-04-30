//
//  CorridorKeyToolboxPlugIn.swift
//  CorridorKey by LateNite
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

    /// Last render backend used by the tile renderer. The inspector header
    /// reads this for the same Backend row the Standalone Editor exposes.
    let lastRenderReport = AtomicRenderReport()

    /// Per-instance analysis session. Holds the in-flight matte cache while a
    /// forward-analysis pass is running; dropped back to empty once cleanup
    /// has flushed the final result to the Library. Owned by the plug-in
    /// (rather than a separate registry) so it deallocates automatically when
    /// Final Cut Pro releases the instance — previously, a process-level
    /// registry leaked a full clip's worth of mattes per instance.
    let analysisSession = AnalysisSessionState()

    /// Last persisted analysis cache for this plug-in instance. The render
    /// callback uses it as a same-process fallback when Final Cut Pro hands
    /// back an older `pluginState` blob after a seek.
    let analysisSnapshotStore = AnalysisSnapshotStore()

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
        // We warm at the rung the `.automatic` default would resolve to
        // for an HD-baseline clip on this machine. That keeps the eager
        // warm-up appropriate for the host's RAM (1024 on a 32 GB Mac,
        // 1536 on 64 GB, 2048 on 96 GB+) — over-warming at 2048 on
        // smaller machines costs gigabytes of unified memory before the
        // user has even picked a clip.
        Self.kickOffDefaultWarmup()
    }

    /// Non-isolated helper that asks `SharedMLXBridgeRegistry` to warm
    /// the default rung on the system default Metal device. Safe to call
    /// from `init`; the registry handles concurrency + deduping.
    private static func kickOffDefaultWarmup() {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        guard let entry = try? MetalDeviceCache.shared.entry(for: device) else { return }
        // Consult the per-device capability cache so the eager warm-up
        // matches the rung this device will actually use at render time —
        // otherwise we warm 1024 only to discover the cache has lifted us
        // to 1536, then pay a second warm-up the moment the user clicks
        // Analyse Clip.
        let defaultRung = QualityMode.automatic.resolvedInferenceResolution(
            forLongEdge: 1920,
            deviceRegistryID: device.registryID
        )
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

/// Thread-safe one-value store for the last render report published by the
/// render callback. Kept separate from `AtomicDouble` so the report remains a
/// value type and the FxPlug bridge can read it without touching Metal state.
final class AtomicRenderReport: @unchecked Sendable {
    private let lock = NSLock()
    private var value: RenderReport?

    func set(_ newValue: RenderReport) {
        lock.lock()
        value = newValue
        lock.unlock()
    }

    func read() -> RenderReport? {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
