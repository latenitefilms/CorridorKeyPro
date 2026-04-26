//
//  SharedMLXBridgeRegistry.swift
//  CorridorKey by LateNite
//
//  Process-wide cache of warmed `MLXKeyingEngine`s keyed by `(device,
//  rung)`. Before this registry each `InferenceCoordinator` held its own
//  `MLXKeyingEngine`, which meant a project with five clips using
//  Corridor Key paid for five copies of the same `.mlxfn` bridge in
//  memory (≈ 300 MB per copy at 1024). With the registry the bridge
//  loads once per `(device, rung)` and is shared across every plug-in
//  instance for the lifetime of the process.
//
//  The registry also enables eager warm-up from
//  `CorridorKeyToolboxPlugIn.init` — when FxPlug creates a new plug-in
//  instance we kick off a warm-up task into this registry; by the time
//  the user clicks Analyse Clip or starts playback, MLX is already
//  compiled and warm, eliminating the 2–5 s first-play stall.
//
//  Lifetime: engines are held strongly for the process's life. On Apple
//  Silicon the underlying MLX buffers sit in unified memory and are
//  released when the engine is dropped; since FCP can return to the
//  same project repeatedly in a session, keeping engines warm matches
//  user intent. Memory-pressure handling (release least-recently-used
//  engines) is a v1.1 concern.
//

import Foundation
import Metal
#if CORRIDOR_KEY_SPM_MIRROR
import CorridorKeyToolboxLogic
#endif

final class SharedMLXBridgeRegistry: @unchecked Sendable {

    /// Process-wide singleton. The registry is stateless from the
    /// outside — callers just ask for an engine by key.
    static let shared = SharedMLXBridgeRegistry()

    private struct Key: Hashable, Sendable {
        let deviceRegistryID: UInt64
        let rung: Int
    }

    private let lock = NSLock()
    private var engines: [Key: MLXKeyingEngine] = [:]
    /// Backing tasks for in-flight warm-ups. Removed when the task
    /// finishes so subsequent lookups don't chase a dead reference.
    private var warmupTasks: [Key: Task<Void, Never>] = [:]
    /// Last failure message per `(device, rung)`. Cleared on successful
    /// warm-up; surfaced to UI via `warmupStatus`.
    private var warmupFailures: [Key: String] = [:]

    private init() {}

    // MARK: - Public API

    /// Returns the engine immediately if it's already warm. Non-blocking,
    /// used on the render hot path.
    func readyEngine(deviceRegistryID: UInt64, rung: Int) -> MLXKeyingEngine? {
        let key = Key(deviceRegistryID: deviceRegistryID, rung: rung)
        lock.lock()
        defer { lock.unlock() }
        return engines[key]
    }

    /// Blocks the calling thread until the `(device, rung)` engine is
    /// ready, the warm-up has failed, or warm-up is cancelled. Kicks off
    /// the warm-up itself if one isn't already running. Used by the
    /// analyser, which **must** key with MLX — falling through to the
    /// rough-matte fallback for a few frames while MLX warms would
    /// silently mix two engines' output into the cached matte sequence
    /// and produce a low-quality, inconsistent result.
    ///
    /// Returns the engine on success, throws `KeyingInferenceError` on
    /// permanent failure.
    func waitForReady(
        deviceRegistryID: UInt64,
        rung: Int,
        cacheEntry: MetalDeviceCacheEntry,
        pollInterval: TimeInterval = 0.05,
        timeout: TimeInterval = 120
    ) throws -> MLXKeyingEngine {
        let deadline = Date().addingTimeInterval(timeout)
        beginWarmup(deviceRegistryID: deviceRegistryID, rung: rung, cacheEntry: cacheEntry)
        while Date() < deadline {
            switch status(deviceRegistryID: deviceRegistryID, rung: rung) {
            case .ready:
                if let engine = readyEngine(deviceRegistryID: deviceRegistryID, rung: rung) {
                    return engine
                }
                // Status flicked to ready but engine vanished — keep polling.
            case .failed(let message):
                throw KeyingInferenceError.modelUnavailable(message)
            case .cold, .warming:
                Thread.sleep(forTimeInterval: pollInterval)
            }
        }
        throw KeyingInferenceError.modelUnavailable(
            "MLX bridge for \(rung)px did not become ready within \(Int(timeout))s."
        )
    }

    /// Kicks off a background warm-up for `(device, rung)` if one isn't
    /// already running and the engine isn't already warm. Idempotent —
    /// repeated calls return immediately when a task is in flight or the
    /// engine is ready.
    func beginWarmup(
        deviceRegistryID: UInt64,
        rung: Int,
        cacheEntry: MetalDeviceCacheEntry
    ) {
        let key = Key(deviceRegistryID: deviceRegistryID, rung: rung)
        lock.lock()
        if engines[key] != nil || warmupTasks[key] != nil {
            lock.unlock()
            return
        }
        // Clear prior failure so a retry gets a fresh status.
        warmupFailures[key] = nil
        lock.unlock()

        let task = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            await self.runWarmup(key: key, cacheEntry: cacheEntry, rung: rung)
        }
        lock.lock()
        warmupTasks[key] = task
        lock.unlock()
    }

    /// Current warm-up status for the given `(device, rung)`. Used by
    /// the inspector bridge to drive the "Loading neural model…" badge.
    func status(deviceRegistryID: UInt64, rung: Int) -> WarmupStatus {
        let key = Key(deviceRegistryID: deviceRegistryID, rung: rung)
        lock.lock()
        defer { lock.unlock() }
        if let message = warmupFailures[key] {
            return .failed(message)
        }
        if engines[key] != nil {
            return .ready(resolution: rung)
        }
        if warmupTasks[key] != nil {
            return .warming(resolution: rung)
        }
        return .cold
    }

    /// Cancels the in-flight warm-up for `(device, rung)` if any.
    /// Other plug-in instances that were also waiting on the same warm-
    /// up lose their in-flight state too, but they'll simply retry on
    /// their next render request. This is the trade-off of sharing —
    /// individual cancellations don't get fine-grained behaviour.
    func cancelWarmup(deviceRegistryID: UInt64, rung: Int) {
        let key = Key(deviceRegistryID: deviceRegistryID, rung: rung)
        lock.lock()
        let task = warmupTasks[key]
        warmupTasks[key] = nil
        lock.unlock()
        task?.cancel()
    }

    // MARK: - Internal

    private func runWarmup(
        key: Key,
        cacheEntry: MetalDeviceCacheEntry,
        rung: Int
    ) async {
        let engine = MLXKeyingEngine(cacheEntry: cacheEntry)
        guard engine.supports(resolution: rung) else {
            record(failure: "No MLX bridge bundled for \(rung)px", forKey: key)
            return
        }
        do {
            try Task.checkCancellation()
            try await engine.prepare(resolution: rung)
            try Task.checkCancellation()
            store(engine: engine, forKey: key)
            PluginLog.notice("Shared MLX engine ready: \(rung)px on \(cacheEntry.device.name).")
        } catch is CancellationError {
            record(failure: "Warm-up cancelled.", forKey: key)
        } catch {
            record(failure: error.localizedDescription, forKey: key)
            PluginLog.error("Shared MLX warm-up failed: \(error.localizedDescription)")
        }
    }

    private func store(engine: MLXKeyingEngine, forKey key: Key) {
        lock.lock()
        engines[key] = engine
        warmupTasks[key] = nil
        warmupFailures[key] = nil
        lock.unlock()
    }

    private func record(failure message: String, forKey key: Key) {
        lock.lock()
        warmupTasks[key] = nil
        warmupFailures[key] = message
        lock.unlock()
    }
}
