//
//  InferenceCoordinator.swift
//  Corridor Key Toolbox
//
//  Chooses a keying engine and manages warm-up. The render pipeline delegates
//  the opaque "ask a model for a matte" problem to this coordinator so the
//  orchestrator can stay linear and testable.
//
//  MLX is loaded **asynchronously** on a detached Task so the render pipeline
//  is never blocked waiting for the bridge to compile. Until MLX is ready,
//  the coordinator returns the `RoughMatteKeyingEngine` so FCP gets a matte
//  immediately. Once warm-up completes the coordinator switches engines and
//  subsequent frames render with the neural matte.
//
//  A single-frame MLX output cache is layered on top: if the user changes a
//  post-process parameter (output mode, despill, matte sliders) without
//  moving the playhead or swapping screen colour, the coordinator returns
//  the previous MLX result instead of paying for another ~500 ms inference.
//

import Foundation
import Metal
import CoreMedia

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

    /// Enables the MLX bridge. When `true` the coordinator asynchronously
    /// loads the bundled `.mlxfn` that matches the requested resolution and
    /// falls back to `RoughMatteKeyingEngine` on any error.
    private static let mlxEnabled = true

    private let stateLock = NSLock()
    private var roughMatteEngine: RoughMatteKeyingEngine?
    private var mlxEngine: MLXKeyingEngine?
    private var mlxEngineReady: Bool = false
    private var mlxEngineLoading: Bool = false
    private var mlxLoadedResolution: Int = 0
    private var mlxCacheEntryID: ObjectIdentifier?

    /// Single-frame cache of the latest MLX inference. The rough-matte path
    /// writes to freshly-allocated textures every frame so it's cheap enough
    /// to re-run; only the expensive MLX result is worth caching.
    private var cachedMLXKey: InferenceCacheKey?
    private var cachedMLXOutput: KeyingInferenceOutput?

    /// Human-readable backend summary for diagnostics.
    var backendDescription: String {
        stateLock.lock()
        defer { stateLock.unlock() }
        if mlxEngineReady, let mlxEngine {
            return mlxEngine.backendDisplayName
        }
        if let roughMatteEngine {
            return roughMatteEngine.backendDisplayName
        }
        return "Idle"
    }

    /// Runs inference for a single frame. Uses MLX when it's warm, otherwise
    /// synchronously produces a rough matte. MLX warm-up is kicked off in
    /// the background on the first frame so subsequent frames can use it.
    ///
    /// Passes `cacheKey` so we can short-circuit repeated inferences at the
    /// same play-head when only post-process parameters have changed.
    func runInference(
        request: KeyingInferenceRequest,
        cacheEntry: MetalDeviceCacheEntry,
        cacheKey: InferenceCacheKey
    ) throws -> KeyingInferenceOutput {
        if let cached = cachedOutput(for: cacheKey) {
            return cached
        }

        let output = try makeOutputTextures(request: request, cacheEntry: cacheEntry)

        // If the resolution has changed since the last warm-up we'll need a
        // different bridge file. Invalidate and re-kick the warm-up.
        if Self.mlxEnabled {
            kickOffMLXWarmupIfNeeded(
                resolution: request.inferenceResolution,
                cacheEntry: cacheEntry
            )
        }

        if let mlx = readyMLXEngine(for: request.inferenceResolution, cacheEntry: cacheEntry) {
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

        let fallback = getOrCreateRoughMatteEngine(cacheEntry: cacheEntry)
        try fallback.run(request: request, output: output)
        // The fallback path intentionally isn't cached — it's already cheap
        // and caching it would starve later MLX-ready frames of a valid slot.
        return output
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

    // MARK: - MLX lifecycle

    /// Returns a ready MLX engine for the given resolution, or nil if warm-up
    /// is still in flight. Checking is cheap — this is called per frame.
    private func readyMLXEngine(
        for resolution: Int,
        cacheEntry: MetalDeviceCacheEntry
    ) -> MLXKeyingEngine? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard mlxEngineReady,
              let mlx = mlxEngine,
              mlxLoadedResolution == resolution,
              mlxCacheEntryID == ObjectIdentifier(cacheEntry)
        else {
            return nil
        }
        return mlx
    }

    /// Starts (or re-starts) an MLX warm-up on a background Task if one isn't
    /// already in flight for the current (resolution, device) combination.
    private func kickOffMLXWarmupIfNeeded(
        resolution: Int,
        cacheEntry: MetalDeviceCacheEntry
    ) {
        stateLock.lock()
        let alreadyLoadingOrLoaded =
            mlxEngineLoading
            || (mlxEngineReady
                && mlxLoadedResolution == resolution
                && mlxCacheEntryID == ObjectIdentifier(cacheEntry))
        if alreadyLoadingOrLoaded {
            stateLock.unlock()
            return
        }
        // Resolution or device changed — reset and start a fresh warm-up.
        mlxEngineReady = false
        mlxEngineLoading = true
        mlxEngine = nil
        mlxLoadedResolution = resolution
        mlxCacheEntryID = ObjectIdentifier(cacheEntry)
        cachedMLXKey = nil
        cachedMLXOutput = nil
        stateLock.unlock()

        let engine = MLXKeyingEngine(cacheEntry: cacheEntry)
        guard engine.supports(resolution: resolution) else {
            PluginLog.notice("No MLX bridge bundled for \(resolution)px; rough matte will be used.")
            stateLock.lock()
            mlxEngineLoading = false
            stateLock.unlock()
            return
        }

        PluginLog.notice("Queuing MLX warm-up for \(resolution)px on \(cacheEntry.device.name).")
        let cacheEntryID = ObjectIdentifier(cacheEntry)
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                try await engine.prepare(resolution: resolution)
                self?.recordMLXWarmupSuccess(engine: engine, resolution: resolution, cacheEntryID: cacheEntryID)
                PluginLog.notice("MLX warm-up complete for \(resolution)px.")
            } catch {
                self?.recordMLXWarmupFailure()
                PluginLog.error("MLX warm-up failed: \(error.localizedDescription). Rough matte will continue to be used.")
            }
        }
    }

    private func recordMLXWarmupSuccess(
        engine: MLXKeyingEngine,
        resolution: Int,
        cacheEntryID: ObjectIdentifier
    ) {
        stateLock.lock()
        mlxEngineLoading = false
        mlxEngine = engine
        mlxEngineReady = true
        mlxLoadedResolution = resolution
        mlxCacheEntryID = cacheEntryID
        cachedMLXKey = nil
        cachedMLXOutput = nil
        stateLock.unlock()
    }

    private func recordMLXWarmupFailure() {
        stateLock.lock()
        mlxEngineLoading = false
        mlxEngine = nil
        mlxEngineReady = false
        cachedMLXKey = nil
        cachedMLXOutput = nil
        stateLock.unlock()
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
