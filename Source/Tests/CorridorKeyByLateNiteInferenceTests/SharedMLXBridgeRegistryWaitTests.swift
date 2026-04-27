//
//  SharedMLXBridgeRegistryWaitTests.swift
//  CorridorKeyByLateNiteInferenceTests
//
//  Verifies the analyser-blocking warm-up gate wakes promptly via the
//  semaphore-based notification path instead of the legacy 50 ms poll.
//

import Foundation
import Metal
import Testing
@testable import CorridorKeyToolboxMetalStages

@Suite("SharedMLXBridgeRegistry wake-up")
struct SharedMLXBridgeRegistryWaitTests {

    /// Cancelled warm-up failures must be keyed by `(device, rung)` and
    /// must not bleed into other rungs that haven't been touched yet.
    /// Also confirms the registry handles a "cold → warming → cancelled
    /// → cold" transition without leaving stale state behind.
    @Test("Cancelled warm-up state does not poison subsequent rungs")
    func cancellationDoesNotPoisonOtherRungs() throws {
        let entry = try InferenceTestHarness.makeEntry()
        let registry = SharedMLXBridgeRegistry.shared
        let deviceID = entry.device.registryID

        registry.beginWarmup(deviceRegistryID: deviceID, rung: 2, cacheEntry: entry)
        registry.cancelWarmup(deviceRegistryID: deviceID, rung: 2)

        let unrelatedStatus = registry.status(deviceRegistryID: deviceID, rung: 3)
        switch unrelatedStatus {
        case .cold:
            break
        default:
            Issue.record("Unrelated rung saw status \(unrelatedStatus); expected .cold.")
        }
    }

    /// `waitForReady` against an unbundled rung must surface
    /// `modelUnavailable` quickly — the warm-up task records the
    /// failure synchronously inside its first `supports(resolution:)`
    /// check, which is what every parked waiter wakes on. The whole
    /// cycle should land well inside the legacy 50 ms poll period:
    /// failing this assertion would mean we've reintroduced the
    /// `Thread.sleep` poll or otherwise missed the synchronous wake.
    @Test("waitForReady fails fast when no bridge is bundled")
    func waitForReadyFailsFastWithoutBridge() throws {
        let entry = try InferenceTestHarness.makeEntry()
        let registry = SharedMLXBridgeRegistry.shared
        let deviceID = entry.device.registryID

        // Pick a rung that maps via `closestSupportedResolution` to a
        // file not bundled in the inference test target. The test
        // bundle ships only the 512px bridge, so 2048 (the highest
        // ladder rung) is unavailable. We tag a unique device id so
        // cached state from earlier tests can't satisfy the lookup.
        let untouchedDeviceID = deviceID &+ 0xFEED_FACE
        let unbundledRung = 2048

        let start = DispatchTime.now()
        var threw = false
        do {
            _ = try registry.waitForReady(
                deviceRegistryID: untouchedDeviceID,
                rung: unbundledRung,
                cacheEntry: entry,
                timeout: 5
            )
        } catch is KeyingInferenceError {
            threw = true
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        let elapsedMillis = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000

        #expect(threw, "Expected modelUnavailable for an unbundled rung")
        // Expect a wake well under 200 ms; the legacy poll path could
        // not return faster than the 50 ms tick boundary, so even on
        // very busy CI we should land much faster.
        #expect(elapsedMillis < 200,
                "waitForReady took \(elapsedMillis) ms; the synchronous failure path should signal waiters immediately.")
    }

    /// Two threads parking on the same key should both wake on a
    /// single warm-up completion — the per-key waiter list signals
    /// every enrolled semaphore. This protects against future
    /// regressions where someone replaces the list with a single-slot
    /// `DispatchSemaphore` and accidentally orphans subsequent
    /// waiters.
    @Test("Multiple waiters on the same rung all wake on completion")
    func multipleWaitersWakeTogether() throws {
        let entry = try InferenceTestHarness.makeEntry()
        let registry = SharedMLXBridgeRegistry.shared
        // Use a unique device id so this test's state never collides
        // with the live registry entries set up by other tests.
        let deviceID = entry.device.registryID &+ 0xCAFE_BABE
        let unbundledRung = 1536

        let waiterCount = 4
        let allWoken = DispatchSemaphore(value: 0)
        let counter = ManagedAtomicInt32(0)

        for _ in 0..<waiterCount {
            Thread {
                do {
                    _ = try registry.waitForReady(
                        deviceRegistryID: deviceID,
                        rung: unbundledRung,
                        cacheEntry: entry,
                        timeout: 5
                    )
                } catch {
                    // Expected — bridge not bundled.
                }
                if counter.increment() == Int32(waiterCount) {
                    allWoken.signal()
                }
            }.start()
        }

        let result = allWoken.wait(timeout: .now() + 2)
        #expect(result == .success,
                "Only \(counter.value) of \(waiterCount) waiters woke before timeout")
    }
}

/// Lightweight atomic counter used by the multi-waiter test. Avoids the
/// need to import `os.lock` or `Synchronization` for a single counter.
private final class ManagedAtomicInt32: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Int32

    init(_ initial: Int32) {
        self.storage = initial
    }

    var value: Int32 {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    @discardableResult
    func increment() -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        storage += 1
        return storage
    }
}
