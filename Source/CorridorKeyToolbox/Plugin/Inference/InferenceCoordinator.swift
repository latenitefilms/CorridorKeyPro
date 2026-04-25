//
//  InferenceCoordinator.swift
//  Corridor Key Toolbox
//
//  Chooses a keying engine, manages the per-frame cache, and falls back
//  to the rough-matte engine when MLX isn't ready yet. Engines live in
//  `SharedMLXBridgeRegistry` — the coordinator no longer owns them —
//  so multiple plug-in instances share a single warmed bridge per
//  `(device, rung)`.
//
//  A single-frame MLX output cache is layered on top: if the user
//  changes a post-process parameter (output mode, despill, matte
//  sliders) without moving the playhead or swapping screen colour, the
//  coordinator returns the previous MLX result instead of paying for
//  another ~500 ms inference.
//

import Foundation
import Metal
import CoreMedia
#if CORRIDOR_KEY_SPM_MIRROR
import CorridorKeyToolboxLogic
#endif

/// Cache key for the single-frame MLX output cache. Anything that changes the
/// neural model's input must participate in the hash; anything purely
/// post-process (output mode, despill strength, etc.) must not.
struct InferenceCacheKey: Hashable, Sendable {
    let frameTimeValue: Int64
    let frameTimeTimescale: Int32
    let screenColorRaw: Int
    let inferenceResolution: Int
    let cacheEntryID: ObjectIdentifier

    init(
        frameTime: CMTime,
        screenColorRaw: Int,
        inferenceResolution: Int,
        cacheEntry: MetalDeviceCacheEntry
    ) {
        self.frameTimeValue = frameTime.value
        self.frameTimeTimescale = frameTime.timescale
        self.screenColorRaw = screenColorRaw
        self.inferenceResolution = inferenceResolution
        self.cacheEntryID = ObjectIdentifier(cacheEntry)
    }
}

final class InferenceCoordinator: @unchecked Sendable {

    /// Enables the MLX bridge. When `true` the coordinator asks the
    /// shared registry to warm a bridge matching the requested rung;
    /// falls back to `RoughMatteKeyingEngine` on any error.
    private static let mlxEnabled = true

    private let stateLock = NSLock()
    private var roughMatteEngine: RoughMatteKeyingEngine?

    /// The `(device, rung)` this coordinator most recently asked the
    /// registry to warm. Used by `warmupStatus` and `cancelWarmup` to
    /// target the right key.
    private var trackedDeviceRegistryID: UInt64 = 0
    private var trackedRung: Int = 0

    /// Single-frame cache of the latest MLX inference. The rough-matte path
    /// writes to freshly-allocated textures every frame so it's cheap enough
    /// to re-run; only the expensive MLX result is worth caching.
    private var cachedMLXKey: InferenceCacheKey?
    private var cachedMLXOutput: KeyingInferenceOutput?

    /// Human-readable backend summary for diagnostics.
    var backendDescription: String {
        stateLock.lock()
        let device = trackedDeviceRegistryID
        let rung = trackedRung
        let rough = roughMatteEngine
        stateLock.unlock()
        if rung > 0, let engine = SharedMLXBridgeRegistry.shared.readyEngine(deviceRegistryID: device, rung: rung) {
            return engine.backendDisplayName
        }
        if let rough {
            return rough.backendDisplayName
        }
        return "Idle"
    }

    /// Current warm-up status. The inspector bridge consumes this to render
    /// the "Loading neural model…" badge during the first-play stall.
    var warmupStatus: WarmupStatus {
        stateLock.lock()
        let device = trackedDeviceRegistryID
        let rung = trackedRung
        stateLock.unlock()
        guard rung > 0 else { return .cold }
        return SharedMLXBridgeRegistry.shared.status(deviceRegistryID: device, rung: rung)
    }

    /// Cancels the in-flight MLX warm-up tracked by this coordinator.
    /// Safe when no warm-up is running — it's a no-op in that case.
    func cancelWarmup() {
        stateLock.lock()
        let device = trackedDeviceRegistryID
        let rung = trackedRung
        stateLock.unlock()
        guard rung > 0 else { return }
        SharedMLXBridgeRegistry.shared.cancelWarmup(deviceRegistryID: device, rung: rung)
    }

    /// Runs inference for a single frame. Uses MLX when the shared
    /// registry has a warm engine for the requested `(device, rung)`;
    /// otherwise synchronously produces a rough matte. Warm-up is
    /// requested in the background so subsequent frames can use MLX.
    ///
    /// Passes `cacheKey` so we can short-circuit repeated inferences at
    /// the same play-head when only post-process parameters have
    /// changed.
    func runInference(
        request: KeyingInferenceRequest,
        cacheEntry: MetalDeviceCacheEntry,
        cacheKey: InferenceCacheKey
    ) throws -> KeyingInferenceOutput {
        if let cached = cachedOutput(for: cacheKey) {
            return cached
        }

        let output = try makeOutputTextures(request: request, cacheEntry: cacheEntry)

        if Self.mlxEnabled {
            let deviceRegistryID = cacheEntry.device.registryID
            trackRequestedBridge(deviceRegistryID: deviceRegistryID, rung: request.inferenceResolution)
            SharedMLXBridgeRegistry.shared.beginWarmup(
                deviceRegistryID: deviceRegistryID,
                rung: request.inferenceResolution,
                cacheEntry: cacheEntry
            )

            if let mlx = SharedMLXBridgeRegistry.shared.readyEngine(
                deviceRegistryID: deviceRegistryID,
                rung: request.inferenceResolution
            ) {
                do {
                    try mlx.run(request: request, output: output)
                    storeCachedOutput(output, for: cacheKey)
                    return output
                } catch {
                    PluginLog.error(
                        "MLX inference failed; using rough matte for this frame. Error: \(error.localizedDescription)"
                    )
                    // Intentional fall-through to the rough-matte path below.
                }
            }
        }

        let fallback = getOrCreateRoughMatteEngine(cacheEntry: cacheEntry)
        try fallback.run(request: request, output: output)
        // The fallback path intentionally isn't cached — it's already cheap
        // and caching it would starve later MLX-ready frames of a valid slot.
        return output
    }

    /// Asks the shared registry to start warm-up for `(device, rung)`
    /// without waiting. Used by `CorridorKeyToolboxPlugIn.init` to
    /// pre-warm the default bridge the moment the effect is applied.
    func requestEagerWarmup(rung: Int, cacheEntry: MetalDeviceCacheEntry) {
        guard Self.mlxEnabled else { return }
        let deviceRegistryID = cacheEntry.device.registryID
        trackRequestedBridge(deviceRegistryID: deviceRegistryID, rung: rung)
        SharedMLXBridgeRegistry.shared.beginWarmup(
            deviceRegistryID: deviceRegistryID,
            rung: rung,
            cacheEntry: cacheEntry
        )
    }

    // MARK: - MLX frame cache

    private func cachedOutput(for key: InferenceCacheKey) -> KeyingInferenceOutput? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard cachedMLXKey == key, let cached = cachedMLXOutput else {
            return nil
        }
        return cached
    }

    private func storeCachedOutput(_ output: KeyingInferenceOutput, for key: InferenceCacheKey) {
        stateLock.lock()
        cachedMLXKey = key
        cachedMLXOutput = output
        stateLock.unlock()
    }

    // MARK: - Tracked bridge target

    private func trackRequestedBridge(deviceRegistryID: UInt64, rung: Int) {
        stateLock.lock()
        if trackedDeviceRegistryID != deviceRegistryID || trackedRung != rung {
            cachedMLXKey = nil
            cachedMLXOutput = nil
        }
        trackedDeviceRegistryID = deviceRegistryID
        trackedRung = rung
        stateLock.unlock()
    }

    // MARK: - Output textures

    private func makeOutputTextures(
        request: KeyingInferenceRequest,
        cacheEntry: MetalDeviceCacheEntry
    ) throws -> KeyingInferenceOutput {
        guard let alpha = cacheEntry.makeIntermediateTexture(
            width: request.inferenceResolution,
            height: request.inferenceResolution,
            pixelFormat: .r32Float,
            storageMode: .shared
        ) else {
            throw KeyingInferenceError.deviceUnavailable
        }
        guard let foreground = cacheEntry.makeIntermediateTexture(
            width: request.inferenceResolution,
            height: request.inferenceResolution,
            pixelFormat: .rgba32Float,
            storageMode: .shared
        ) else {
            throw KeyingInferenceError.deviceUnavailable
        }
        return KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)
    }

    // MARK: - Rough matte fallback

    private func getOrCreateRoughMatteEngine(cacheEntry: MetalDeviceCacheEntry) -> RoughMatteKeyingEngine {
        stateLock.lock()
        if let existing = roughMatteEngine {
            stateLock.unlock()
            return existing
        }
        stateLock.unlock()
        let created = RoughMatteKeyingEngine(cacheEntry: cacheEntry)
        stateLock.lock()
        roughMatteEngine = created
        stateLock.unlock()
        return created
    }
}
