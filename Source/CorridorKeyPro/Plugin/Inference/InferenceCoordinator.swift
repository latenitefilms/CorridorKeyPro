//
//  InferenceCoordinator.swift
//  Corridor Key Pro
//
//  Chooses a keying engine, manages warm-up, and recovers from engine-level
//  failures. The render pipeline delegates the opaque "ask a model for a matte"
//  problem to this coordinator so the orchestrator can stay linear and testable.
//

import Foundation
import Metal

/// Wraps the per-instance engine selection and warm-up. Thread-safe so
/// Final Cut Pro can drive multiple tiles into the same instance at once.
final class InferenceCoordinator: @unchecked Sendable {
    private let stateLock = NSLock()
    private var currentEngine: KeyingInferenceEngine?
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

        // Output textures are allocated with `.shared` storage so the engine
        // can populate them from the CPU via `texture.replace(region:…)` after
        // inference, and the post-inference GPU pass can read them back.
        guard let alpha = cacheEntry.makeIntermediateTexture(
            width: request.inferenceResolution,
            height: request.inferenceResolution,
            pixelFormat: .r16Float,
            storageMode: .shared
        ) else {
            throw KeyingInferenceError.deviceUnavailable
        }
        guard let foreground = cacheEntry.makeIntermediateTexture(
            width: request.inferenceResolution,
            height: request.inferenceResolution,
            pixelFormat: .rgba16Float,
            storageMode: .shared
        ) else {
            throw KeyingInferenceError.deviceUnavailable
        }
        let output = KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)

        do {
            try engine.run(request: request, output: output)
            return output
        } catch {
            // Fall back to the rough-matte engine on any error so the render
            // always completes; the status field will reflect the fallback.
            let fallback = RoughMatteKeyingEngine(cacheEntry: cacheEntry)
            try fallback.run(request: request, output: output)
            stateLock.lock()
            currentEngine = fallback
            stateLock.unlock()
            return output
        }
    }

    /// Returns the engine that should service the given resolution, creating
    /// it lazily on first use.
    private func engine(
        for resolution: Int,
        cacheEntry: MetalDeviceCacheEntry
    ) throws -> KeyingInferenceEngine {
        stateLock.lock()
        if let engine = currentEngine,
           engine.supports(resolution: resolution),
           warmResolution == resolution,
           warmCacheEntryID == ObjectIdentifier(cacheEntry) {
            stateLock.unlock()
            return engine
        }
        stateLock.unlock()

        let preferred = MLXKeyingEngine(cacheEntry: cacheEntry)
        if preferred.supports(resolution: resolution) {
            do {
                // Synchronously wait for the MLX bridge to warm up. The first
                // call loads the `.mlxfn` file and JIT-compiles the graph;
                // subsequent calls reuse the cached function handle.
                try runBlocking { try await preferred.prepare(resolution: resolution) }
                stateLock.lock()
                currentEngine = preferred
                warmResolution = resolution
                warmCacheEntryID = ObjectIdentifier(cacheEntry)
                stateLock.unlock()
                return preferred
            } catch {
                // Fall through to the rough matte engine.
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
    /// caller until it completes. Required because FxPlug's render entry
    /// point is itself synchronous; Final Cut Pro manages concurrency above us.
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

/// Minimal thread-safe storage for a `Result` value produced by a background
/// task and read from the thread that kicked it off.
private final class RunBlockingBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<T, Error> = .failure(KeyingInferenceError.deviceUnavailable)

    func set(_ value: Result<T, Error>) {
        lock.lock(); result = value; lock.unlock()
    }

    func get() throws -> T {
        lock.lock(); defer { lock.unlock() }
        return try result.get()
    }
}
