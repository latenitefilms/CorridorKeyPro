//
//  InferenceCoordinator.swift
//  Corridor Key Pro
//
//  Chooses a keying engine and manages warm-up. The render pipeline delegates
//  the opaque "ask a model for a matte" problem to this coordinator so the
//  orchestrator can stay linear and testable.
//
//  MLX integration is gated behind `Self.mlxEnabled`. MLX is currently
//  disabled so the plug-in has a fast, dependency-free render path while the
//  neural bridge is validated against CorridorKey's canonical outputs. When
//  MLX is turned back on, the coordinator will prefer it and fall back to
//  the rough-matte engine on any failure.
//

import Foundation
import Metal

final class InferenceCoordinator: @unchecked Sendable {

    /// Set to `true` once the MLX bridge's input/output convention is
    /// validated end-to-end against CorridorKey's reference OFX plug-in.
    /// Currently off because the bundled bridge produces an inverted matte
    /// at runtime — the pipeline gets the same artefact every frame whether
    /// MLX is involved or not, so we ship the deterministic rough-matte
    /// path until we have the right pre/post-processing wired up.
    private static let mlxEnabled = false

    private let stateLock = NSLock()
    private var currentEngine: (any KeyingInferenceEngine)?
    private var warmResolution: Int = 0
    private var warmCacheEntryID: ObjectIdentifier?

    /// Human-readable backend summary surfaced in the Runtime status panel.
    var backendDescription: String {
        stateLock.lock()
        defer { stateLock.unlock() }
        return currentEngine?.backendDisplayName ?? "Idle"
    }

    /// Runs inference for a single frame. Responsible for creating the engine
    /// on first use and for graceful downgrade when the preferred engine can't
    /// honour the current request.
    func runInference(
        request: KeyingInferenceRequest,
        cacheEntry: MetalDeviceCacheEntry
    ) throws -> KeyingInferenceOutput {
        let engine = try engine(for: request.inferenceResolution, cacheEntry: cacheEntry)

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
        let output = KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)

        do {
            try engine.run(request: request, output: output)
            return output
        } catch {
            PluginLog.error(
                "Inference engine \(engine.backendDisplayName) failed; falling back. Error: \(error.localizedDescription)"
            )
            let fallback = RoughMatteKeyingEngine(cacheEntry: cacheEntry)
            try fallback.run(request: request, output: output)
            stateLock.lock()
            currentEngine = fallback
            stateLock.unlock()
            return output
        }
    }

    private func engine(
        for resolution: Int,
        cacheEntry: MetalDeviceCacheEntry
    ) throws -> any KeyingInferenceEngine {
        stateLock.lock()
        if let engine = currentEngine,
           engine.supports(resolution: resolution),
           warmResolution == resolution,
           warmCacheEntryID == ObjectIdentifier(cacheEntry) {
            stateLock.unlock()
            return engine
        }
        stateLock.unlock()

        if Self.mlxEnabled {
            let preferred = MLXKeyingEngine(cacheEntry: cacheEntry)
            if preferred.supports(resolution: resolution) {
                do {
                    try runBlocking { try await preferred.prepare(resolution: resolution) }
                    stateLock.lock()
                    currentEngine = preferred
                    warmResolution = resolution
                    warmCacheEntryID = ObjectIdentifier(cacheEntry)
                    stateLock.unlock()
                    PluginLog.notice("MLX engine ready for \(resolution)px inference on \(cacheEntry.device.name).")
                    return preferred
                } catch {
                    PluginLog.error("MLX warm-up failed; using rough matte fallback. Error: \(error.localizedDescription)")
                }
            } else {
                PluginLog.notice("No MLX bridge bundled for \(resolution)px; using rough matte fallback.")
            }
        }

        let fallback = RoughMatteKeyingEngine(cacheEntry: cacheEntry)
        stateLock.lock()
        currentEngine = fallback
        warmResolution = resolution
        warmCacheEntryID = ObjectIdentifier(cacheEntry)
        stateLock.unlock()
        return fallback
    }

    /// Runs an async throwing closure on a detached task and blocks the
    /// caller until it completes. FxPlug's render entry point is synchronous,
    /// so we bridge Swift concurrency back to the render thread for MLX
    /// warm-up only.
    private func runBlocking<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T where T: Sendable {
        let semaphore = DispatchSemaphore(value: 0)
        let resultBox = RunBlockingBox<T>()
        Task.detached {
            do {
                let value = try await body()
                resultBox.set(.success(value))
            } catch {
                resultBox.set(.failure(error))
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try resultBox.get()
    }
}

private final class RunBlockingBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<T, any Error> = .failure(KeyingInferenceError.deviceUnavailable)

    func set(_ value: Result<T, any Error>) {
        lock.lock(); result = value; lock.unlock()
    }

    func get() throws -> T {
        lock.lock(); defer { lock.unlock() }
        return try result.get()
    }
}
