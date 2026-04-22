//
//  CorridorKeyProPlugIn.swift
//  Corridor Key Pro
//
//  Main FxPlug entry point. Final Cut Pro instantiates this class through the
//  `ProPlugPlugInList` declaration in Info.plist and routes every tileable
//  effect call, analysis call, and UI callback through it. Keeping the class
//  thin and delegating work to dedicated helpers keeps the integration layer
//  easy to maintain.
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
    let apiManager: PROAPIAccessing

    /// Lock that guards analyser scratch state. Final Cut Pro may call analyser
    /// and render methods from different threads so this is explicitly a lock
    /// rather than an actor.
    let analysisLock = NSLock()

    /// Per-instance renderer. Lazy so that instantiation in `addParameters` is
    /// cheap and so the first touch of Metal happens in the render thread.
    lazy var renderPipeline: RenderPipeline = RenderPipeline()

    /// Rolling timing of the last rendered frame in milliseconds. Used to
    /// populate the "Last Frame" runtime status field.
    let lastFrameMilliseconds = AtomicDouble(0)

    /// Frames produced by the analysis pass, one per timeline frame the user
    /// asked us to analyse. Persisted via the hidden custom analysis parameter.
    var analyzedFrames: [AnalyzedFrame] = []
    var analysisFrameDuration: CMTime = .invalid

    struct AnalyzedFrame: Codable, Sendable {
        let frameTimeValue: Int64
        let frameTimescale: Int32
        let screenReferenceR: Float
        let screenReferenceG: Float
        let screenReferenceB: Float
        let estimatedDifficulty: Double

        init(frameTime: CMTime, screenReference: SIMD3<Float>, estimatedDifficulty: Double) {
            self.frameTimeValue = frameTime.value
            self.frameTimescale = frameTime.timescale
            self.screenReferenceR = screenReference.x
            self.screenReferenceG = screenReference.y
            self.screenReferenceB = screenReference.z
            self.estimatedDifficulty = estimatedDifficulty
        }

        var frameTime: CMTime {
            CMTime(value: frameTimeValue, timescale: frameTimescale)
        }
        var screenReference: SIMD3<Float> {
            SIMD3<Float>(screenReferenceR, screenReferenceG, screenReferenceB)
        }
    }

    // MARK: - Init

    @objc required init?(apiManager: PROAPIAccessing) {
        self.apiManager = apiManager
        super.init()
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
