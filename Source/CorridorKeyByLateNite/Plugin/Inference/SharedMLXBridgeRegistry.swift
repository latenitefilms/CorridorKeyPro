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

    /// Per-key list of waiters that should be woken when warm-up
    /// finishes (success or failure). Each waiter holds a
    /// `DispatchSemaphore` so the analyser thread can park without
    /// burning CPU on a 50 ms poll loop.
    private final class WaiterList {
        var semaphores: [DispatchSemaphore] = []
    }

    private let lock = NSLock()
    private var engines: [Key: MLXKeyingEngine] = [:]
    /// Backing tasks for in-flight warm-ups. Removed when the task
    /// finishes so subsequent lookups don't chase a dead reference.
    private var warmupTasks: [Key: Task<Void, Never>] = [:]
    /// Last failure message per `(device, rung)`. Cleared on successful
    /// warm-up; surfaced to UI via `warmupStatus`.
    private var warmupFailures: [Key: String] = [:]
    /// Threads parked in `waitForReady` waiting for `(device, rung)` to
    /// finish warming up. Signalled by `store(...)` / `record(...)` so
    /// the analyser wakes the moment warm-up completes instead of on
    /// the next poll tick.
    private var waiters: [Key: WaiterList] = [:]

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
    /// Implementation: enrolls a `DispatchSemaphore` on the per-key
    /// waiter list before checking status, then parks on the semaphore
    /// until the warm-up task signals it (success or failure). This
    /// avoids the legacy 50 ms `Thread.sleep` poll loop — wake latency
    /// is now bounded by GCD's signal delivery (<1 ms) instead of the
    /// poll interval, and the analyser thread is blocked, not spinning.
    ///
    /// Returns the engine on success, throws `KeyingInferenceError` on
    /// permanent failure.
    func waitForReady(
        deviceRegistryID: UInt64,
        rung: Int,
        cacheEntry: MetalDeviceCacheEntry,
        timeout: TimeInterval = 120
    ) throws -> MLXKeyingEngine {
        let key = Key(deviceRegistryID: deviceRegistryID, rung: rung)
        beginWarmup(deviceRegistryID: deviceRegistryID, rung: rung, cacheEntry: cacheEntry)
        let deadline = DispatchTime.now() + timeout

        while true {
            // Enroll a fresh waiter under the lock. The same lock guards
            // status mutations in `store(...)` / `record(...)`, so any
            // status change after this point is guaranteed to either be
            // observed below or to signal the semaphore.
            let semaphore = DispatchSemaphore(value: 0)
            lock.lock()
            if let engine = engines[key] {
                lock.unlock()
                return engine
            }
            if let message = warmupFailures[key] {
                lock.unlock()
                throw KeyingInferenceError.modelUnavailable(message)
            }
            // Warm-up is in flight — register and wait.
            let list = waiters[key] ?? WaiterList()
            list.semaphores.append(semaphore)
            waiters[key] = list
            lock.unlock()

            let result = semaphore.wait(timeout: deadline)
            if result == .timedOut {
                // Pull the semaphore back out so a delayed signal
                // doesn't accidentally unblock a future caller.
                detachWaiter(semaphore, forKey: key)
                throw KeyingInferenceError.modelUnavailable(
                    "MLX bridge for \(rung)px did not become ready within \(Int(timeout))s."
                )
            }
            // Loop back to read state under the lock — covers the rare
            // case where the warm-up task was cancelled and rescheduled
            // between the signal and our wake-up.
        }
    }

    /// Removes `semaphore` from the waiter list for `key`. Called when
    /// `waitForReady` times out so a stray late signal can't unblock the
    /// next caller's semaphore by accident. Safe to call when the
    /// semaphore was already removed by a successful wake-up.
    private func detachWaiter(_ semaphore: DispatchSemaphore, forKey key: Key) {
        lock.lock()
        defer { lock.unlock() }
        guard let list = waiters[key] else { return }
        list.semaphores.removeAll { $0 === semaphore }
        if list.semaphores.isEmpty {
            waiters.removeValue(forKey: key)
        }
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

        // Priority is `.userInitiated`, not `.utility`, so a
        // synchronous `waitForReady` caller — usually the analyser
        // thread, which itself runs at user-initiated QoS — never
        // blocks a higher-priority thread on lower-priority work.
        // macOS surfaces that pattern as a priority-inversion
        // warning the moment a `DispatchSemaphore.wait` parks the
        // analyse thread. Eager warm-up (no caller waiting) is
        // also a one-shot per `(device, rung)`, so bumping the
        // priority doesn't inflate steady-state CPU usage either.
        let task = Task.detached(priority: .userInitiated) { [weak self] in
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
        let parked = waiters.removeValue(forKey: key)?.semaphores ?? []
        lock.unlock()
        // Wake every parked analyser thread. Calling `signal()` outside
        // the lock keeps the wake path fast and prevents a waiter that
        // immediately re-enters `waitForReady` from contending with us.
        parked.forEach { $0.signal() }
    }

    private func record(failure message: String, forKey key: Key) {
        lock.lock()
        warmupTasks[key] = nil
        warmupFailures[key] = message
        let parked = waiters.removeValue(forKey: key)?.semaphores ?? []
        lock.unlock()
        parked.forEach { $0.signal() }
    }
}
