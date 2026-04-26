//
//  DeviceCapabilityCache.swift
//  CorridorKey by LateNite
//
//  Learning cache that records per-device MLX inference latency at each
//  rung and lets `QualityMode.automatic` lift the static RAM-aware
//  ceiling on hardware that's empirically fast enough.
//
//  How it works:
//
//  * Every time `MLXKeyingEngine.run` finishes, it calls
//    `record(deviceRegistryID:rung:milliseconds:)` with the wall time.
//  * The cache keeps a rolling median (over the last `sampleHistory`
//    samples) per `(device, rung)` and persists those medians to
//    `UserDefaults` so subsequent app launches benefit.
//  * `recommendedCeiling(for:physicalMemoryBytes:)` walks the rungs
//    upward from the static RAM ceiling and returns the highest rung
//    whose median latency stays below `realtimeBudgetMilliseconds`.
//
//  This is deliberately conservative: we only bump the ceiling when the
//  user has actually run inference at the bumped rung at least once
//  (i.e. by manually picking it), so the cache never invents capability
//  it hasn't observed. New users see the conservative default; users
//  who once tried `Maximum` and found it fast will see the bump
//  reflected in `Recommended` on subsequent sessions.
//

import Foundation

/// Persisted, thread-safe per-device latency cache. Singleton because
/// the data is process-wide and Final Cut Pro can have many plug-in
/// instances reporting into the same cache.
public final class DeviceCapabilityCache: @unchecked Sendable {

    public static let shared = DeviceCapabilityCache()

    /// Wall-time budget under which a rung is considered real-time
    /// playable on this device. 200 ms is a generous bound — at 30 fps
    /// the per-frame budget is 33 ms, so 200 ms means ~6× slower than
    /// real-time, well inside the "scrubbable while editing" envelope
    /// for a cached matte timeline. We keep it generous because even a
    /// non-real-time rung is still useful for offline analyse.
    public static let realtimeBudgetMilliseconds: Double = 200

    /// Number of samples retained per `(device, rung)`. Larger windows
    /// damp transient outliers (a thermal-throttled inference, a
    /// background process spike) but slow learning.
    public static let sampleHistory: Int = 8

    /// Persistent backing store. `UserDefaults.standard` is shared with
    /// the host app; we namespace under our own bundle identifier so
    /// neither side stomps the other.
    private let userDefaults: UserDefaults

    private let lock = NSLock()
    /// Per-`(deviceRegistryID, rung)` rolling sample buffers. Median
    /// recomputed on demand from this array.
    private var samples: [Key: [Double]] = [:]

    private struct Key: Hashable {
        let deviceRegistryID: UInt64
        let rung: Int
    }

    private static let userDefaultsKey = "com.corridor.keytoolbox.deviceCapabilityCache"

    private init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
        loadFromDisk()
    }

    /// Test seam: dependency-injected cache backed by an isolated
    /// `UserDefaults` so concurrent test runs don't race on the
    /// shared store.
    public init(userDefaultsForTesting: UserDefaults) {
        self.userDefaults = userDefaultsForTesting
        loadFromDisk()
    }

    // MARK: - Recording

    /// Records a single MLX inference wall time. Called from the run
    /// path so every analyse / live-render frame contributes a sample.
    /// Cheap (~one dictionary write); the persistence happens on a
    /// debounce so we don't write to disk on every frame.
    public func record(deviceRegistryID: UInt64, rung: Int, milliseconds: Double) {
        guard milliseconds.isFinite, milliseconds > 0, milliseconds < 60_000 else { return }
        let key = Key(deviceRegistryID: deviceRegistryID, rung: rung)
        lock.lock()
        var window = samples[key] ?? []
        window.append(milliseconds)
        if window.count > Self.sampleHistory {
            window.removeFirst(window.count - Self.sampleHistory)
        }
        samples[key] = window
        lock.unlock()
        scheduleWriteToDisk()
    }

    /// Returns the median of the last `sampleHistory` samples for
    /// `(device, rung)`, or `nil` when the device has never run a
    /// frame at that rung.
    public func medianMilliseconds(deviceRegistryID: UInt64, rung: Int) -> Double? {
        let key = Key(deviceRegistryID: deviceRegistryID, rung: rung)
        lock.lock()
        let window = samples[key] ?? []
        lock.unlock()
        guard !window.isEmpty else { return nil }
        let sorted = window.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) * 0.5
        }
        return sorted[mid]
    }

    // MARK: - Recommendation

    /// Returns the highest rung the device has empirically demonstrated
    /// it can serve under the real-time budget, falling back to
    /// `staticCeiling` when the device has no recorded data above
    /// that ceiling. Walks the supplied ladder bottom-up so a rung
    /// without samples blocks all higher rungs from being recommended
    /// (we never invent capability we haven't observed).
    public func recommendedCeiling(
        deviceRegistryID: UInt64,
        staticCeiling: Int,
        ladder: [Int],
        budgetMilliseconds: Double = realtimeBudgetMilliseconds
    ) -> Int {
        var best = staticCeiling
        for rung in ladder where rung > staticCeiling {
            guard let median = medianMilliseconds(
                deviceRegistryID: deviceRegistryID,
                rung: rung
            ) else {
                // No data at this rung; we never bump past a rung we
                // haven't probed. Stop walking — higher rungs would
                // require crossing this gap.
                break
            }
            if median > budgetMilliseconds {
                break
            }
            best = rung
        }
        return best
    }

    // MARK: - Persistence

    private var pendingWriteWorkItem: DispatchWorkItem?
    private static let writeDebounceQueue = DispatchQueue(
        label: "corridorkey.devicecapabilitycache.write",
        qos: .utility
    )

    /// Debounces disk writes to once per second so a 30 fps analyse
    /// pass doesn't pin the I/O scheduler.
    private func scheduleWriteToDisk() {
        lock.lock()
        pendingWriteWorkItem?.cancel()
        let work = DispatchWorkItem { [weak self] in
            self?.writeToDisk()
        }
        pendingWriteWorkItem = work
        lock.unlock()
        Self.writeDebounceQueue.asyncAfter(deadline: .now() + 1.0, execute: work)
    }

    private func writeToDisk() {
        let snapshot: [Key: [Double]]
        lock.lock()
        snapshot = samples
        lock.unlock()

        var encoded: [String: [Double]] = [:]
        for (key, window) in snapshot {
            encoded["\(key.deviceRegistryID).\(key.rung)"] = window
        }
        userDefaults.set(encoded, forKey: Self.userDefaultsKey)
    }

    private func loadFromDisk() {
        guard let stored = userDefaults.dictionary(forKey: Self.userDefaultsKey) as? [String: [Double]] else {
            return
        }
        var loaded: [Key: [Double]] = [:]
        for (rawKey, window) in stored {
            let parts = rawKey.split(separator: ".")
            guard parts.count == 2,
                  let deviceID = UInt64(parts[0]),
                  let rung = Int(parts[1])
            else { continue }
            loaded[Key(deviceRegistryID: deviceID, rung: rung)] = window
        }
        lock.lock()
        samples = loaded
        lock.unlock()
    }

    /// Test entry point: drops every sample. Persists the empty state
    /// so subsequent runs start from the cleared cache.
    public func clearAll() {
        lock.lock()
        samples.removeAll()
        lock.unlock()
        userDefaults.removeObject(forKey: Self.userDefaultsKey)
    }
}
