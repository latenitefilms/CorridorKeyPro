//
//  CorridorKeyProPlugIn.swift
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

@objc(CorridorKeyProPlugIn)
final class CorridorKeyProPlugIn: NSObject, FxTileableEffect, FxAnalyzer {

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

    // MARK: - Init

    @objc required init?(apiManager: any PROAPIAccessing) {
        self.apiManager = apiManager
        super.init()
        PluginLog.notice("CorridorKeyProPlugIn instantiated.")
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
