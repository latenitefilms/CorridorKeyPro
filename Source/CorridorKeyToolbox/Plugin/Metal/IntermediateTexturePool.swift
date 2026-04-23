//
//  IntermediateTexturePool.swift
//  Corridor Key Toolbox
//
//  Reuses Metal textures between frames instead of allocating fresh ones on
//  every render. Profiling v0 Corridor Key Toolbox renders at 3840 × 2160
//  showed ~8–12 intermediate textures allocated per frame (despill output,
//  two morphology ping-pongs, two Gaussian ping-pongs, resample targets,
//  etc.) — at 30 fps this added up to several hundred texture allocations per
//  second per plug-in instance, which shows up as command-queue kickoff
//  jitter in the Metal system trace.
//
//  Design:
//  * One pool per Metal device (owned by `MetalDeviceCacheEntry`).
//  * Textures are keyed by `(width, height, pixelFormat, usage, storageMode)`.
//    Two sites asking for the same shape and flags collapse onto the same
//    free list so they share the same physical memory over time.
//  * Acquired textures are returned via the command buffer's completion
//    handler, guaranteeing the GPU is done with them before reuse.
//  * A soft cap per-bucket keeps peak memory sane on pathological timelines
//    (e.g. compressed + uncompressed rendered side-by-side in different
//    instances).
//
//  We intentionally do **not** use `MTLHeap`. `MTLHeap` needs explicit hazard
//  tracking with `MTLFence`s when textures flip direction between command
//  buffers, and the bookkeeping complexity outweighs the win on Apple Silicon
//  where the driver aggressively recycles device-allocated textures under
//  unified memory. The benefit here comes from not calling
//  `device.makeTexture` on the render thread; the underlying memory does not
//  move whether a heap holds it or not.
//

import Foundation
import Metal

/// Identity key for a texture shape. Buckets of textures with the same key
/// are interchangeable.
///
/// `MTLTextureUsage` is an `OptionSet` but doesn't participate in Swift's
/// automatic `Hashable` synthesis, so we store its `rawValue` directly and
/// provide explicit hash / equality.
struct TextureDescriptorKey: Hashable, Sendable {
    let width: Int
    let height: Int
    let pixelFormat: MTLPixelFormat
    let usageRawValue: UInt
    let storageMode: MTLStorageMode

    init(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat,
        usage: MTLTextureUsage,
        storageMode: MTLStorageMode
    ) {
        self.width = max(width, 1)
        self.height = max(height, 1)
        self.pixelFormat = pixelFormat
        self.usageRawValue = usage.rawValue
        self.storageMode = storageMode
    }

    /// Reconstitutes the original usage flags for callers that rebuild a
    /// texture descriptor from the key.
    var usage: MTLTextureUsage { MTLTextureUsage(rawValue: usageRawValue) }
}

/// Lightweight wrapper returned by `acquire`. The pool adds a completion
/// handler on the supplied command buffer so callers don't need to remember to
/// return textures manually — the texture returns itself as soon as the GPU
/// lets it go. Direct calls to `returnManually()` are available for code
/// paths that don't go through a command buffer (for example, readback tests
/// in the logic suite).
final class PooledTexture: @unchecked Sendable {
    let texture: any MTLTexture
    private let key: TextureDescriptorKey
    private weak var pool: IntermediateTexturePool?
    private var hasReturned: Bool = false
    private let returnLock = NSLock()

    fileprivate init(
        texture: any MTLTexture,
        key: TextureDescriptorKey,
        pool: IntermediateTexturePool
    ) {
        self.texture = texture
        self.key = key
        self.pool = pool
    }

    /// Register a command buffer so the texture is returned once that buffer
    /// is fully retired on the GPU. Safe to call multiple times — subsequent
    /// registrations no-op once the texture has been returned.
    func returnOnCompletion(of commandBuffer: any MTLCommandBuffer) {
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.returnManually()
        }
    }

    /// Returns the texture to the pool immediately. The caller guarantees the
    /// GPU is no longer reading or writing the texture.
    func returnManually() {
        returnLock.lock()
        let alreadyReturned = hasReturned
        hasReturned = true
        returnLock.unlock()
        if alreadyReturned { return }
        pool?.relinquish(texture: texture, key: key)
    }

    deinit {
        // Guard-rail: if a pooled texture leaks out with no command-buffer
        // return registered and no manual return called, we still recycle it
        // so the pool doesn't bleed memory. Real usage should never rely on
        // this path — `returnOnCompletion(of:)` is the canonical form.
        returnManually()
    }
}

/// Pool of reusable intermediate textures for one Metal device.
final class IntermediateTexturePool: @unchecked Sendable {

    /// Upper bound on the number of textures we keep cached per bucket. Chosen
    /// so that even the heaviest pipeline path (post-inference stage at 4K)
    /// won't evict on a single frame, while still capping drift when the
    /// user scrubs across two timelines of different sizes back-to-back.
    private static let maximumTexturesPerBucket: Int = 6

    private let device: any MTLDevice
    private let bucketsLock = NSLock()
    private var buckets: [TextureDescriptorKey: [any MTLTexture]] = [:]

    /// Debug counter the unit tests consult to confirm reuse is happening.
    private let statsLock = NSLock()
    private var acquisitionHits: Int = 0
    private var acquisitionMisses: Int = 0

    init(device: any MTLDevice) {
        self.device = device
    }

    /// Returns a texture that matches the descriptor. If the pool has a free
    /// texture in the matching bucket, it is handed back; otherwise a new one
    /// is allocated from the device. Returns `nil` only when `device` refuses
    /// to allocate — callers treat that as a fatal render error.
    func acquire(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .rgba16Float,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite],
        storageMode: MTLStorageMode = .private
    ) -> PooledTexture? {
        let key = TextureDescriptorKey(
            width: width,
            height: height,
            pixelFormat: pixelFormat,
            usage: usage,
            storageMode: storageMode
        )

        bucketsLock.lock()
        if var bucket = buckets[key], !bucket.isEmpty {
            let texture = bucket.removeLast()
            buckets[key] = bucket
            bucketsLock.unlock()
            recordHit()
            return PooledTexture(texture: texture, key: key, pool: self)
        }
        bucketsLock.unlock()

        guard let descriptor = Self.makeDescriptor(for: key),
              let texture = device.makeTexture(descriptor: descriptor)
        else {
            return nil
        }
        recordMiss()
        return PooledTexture(texture: texture, key: key, pool: self)
    }

    /// Drops every texture currently held. Used when the device goes away
    /// (host tears the entry down) and from tests that want a clean slate.
    func purge() {
        bucketsLock.lock()
        buckets.removeAll(keepingCapacity: false)
        bucketsLock.unlock()
    }

    /// Reuse statistics. Returns `(hits, misses)` where a hit means a cached
    /// texture was handed back and a miss means a new allocation happened.
    func statistics() -> (hits: Int, misses: Int) {
        statsLock.lock()
        defer { statsLock.unlock() }
        return (acquisitionHits, acquisitionMisses)
    }

    // MARK: - Internal

    fileprivate func relinquish(texture: any MTLTexture, key: TextureDescriptorKey) {
        bucketsLock.lock()
        var bucket = buckets[key, default: []]
        if bucket.count < Self.maximumTexturesPerBucket {
            bucket.append(texture)
            buckets[key] = bucket
        }
        bucketsLock.unlock()
    }

    private func recordHit() {
        statsLock.lock()
        acquisitionHits += 1
        statsLock.unlock()
    }

    private func recordMiss() {
        statsLock.lock()
        acquisitionMisses += 1
        statsLock.unlock()
    }

    private static func makeDescriptor(for key: TextureDescriptorKey) -> MTLTextureDescriptor? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: key.pixelFormat,
            width: key.width,
            height: key.height,
            mipmapped: false
        )
        descriptor.usage = key.usage
        descriptor.storageMode = key.storageMode
        return descriptor
    }
}
