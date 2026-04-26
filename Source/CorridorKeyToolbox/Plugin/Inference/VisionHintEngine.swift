//
//  VisionHintEngine.swift
//  Corridor Key Toolbox
//
//  Generates the alpha-hint texture that feeds the MLX bridge's 4th input
//  channel from Apple's Vision framework instead of the legacy green-bias
//  rough matte. `VNGenerateForegroundInstanceMaskRequest` runs on the
//  Neural Engine and returns a per-instance subject mask — a far better
//  prior than `g - max(r, b)` because it segments by saliency, not by
//  green channel dominance, so skin in low light, foliage, or lit
//  foreground objects don't bleed into the hint.
//
//  Lifecycle notes:
//
//  * The CVMetalTexture returned by `CVMetalTextureCacheCreateTextureFromImage`
//    only keeps its underlying IOSurface alive for as long as the
//    `CVMetalTexture` object lives. Callers MUST hold onto the
//    `VisionMask` value until the command buffer that reads it has
//    retired — `VisionMask.retainOnCompletion(of:)` makes that explicit.
//
//  * Vision's `perform` method is synchronous but fast (~10–30 ms on
//    M-series). It runs on the Neural Engine in parallel with any GPU
//    work the caller has already encoded, so the analyser can pipeline
//    Vision against the screen-matrix pass.
//

import Foundation
import Metal
import CoreVideo
import CoreImage
import Vision

/// Wraps a Vision-generated mask as a Metal texture. The pixel data lives
/// in an IOSurface that the texture aliases; the `cvTexture` and
/// `pixelBuffer` fields keep that surface alive for the GPU read.
struct VisionMask: @unchecked Sendable {
    let texture: any MTLTexture
    private let cvTexture: CVMetalTexture
    private let pixelBuffer: CVPixelBuffer

    init(texture: any MTLTexture, cvTexture: CVMetalTexture, pixelBuffer: CVPixelBuffer) {
        self.texture = texture
        self.cvTexture = cvTexture
        self.pixelBuffer = pixelBuffer
    }

    /// Pins the backing IOSurface to the supplied command buffer's
    /// completion. Without this the surface can be reclaimed before
    /// the GPU finishes reading the texture, producing intermittent
    /// black hints.
    func retainOnCompletion(of commandBuffer: any MTLCommandBuffer) {
        commandBuffer.addCompletedHandler { _ in
            _ = self
        }
    }
}

enum VisionHintError: Error, CustomStringConvertible {
    case textureCacheCreationFailed(OSStatus)
    case requestFailed(any Error)
    case maskGenerationFailed(any Error)
    case textureWrappingFailed(OSStatus)

    var description: String {
        switch self {
        case .textureCacheCreationFailed(let status):
            return "Failed to create CVMetalTextureCache (status=\(status))."
        case .requestFailed(let error):
            return "Vision foreground request failed: \(error.localizedDescription)"
        case .maskGenerationFailed(let error):
            return "Vision mask scaling failed: \(error.localizedDescription)"
        case .textureWrappingFailed(let status):
            return "Failed to wrap Vision mask as Metal texture (status=\(status))."
        }
    }
}

/// Runs Apple's foreground subject detector and returns the result as a
/// Metal texture suitable for feeding the existing `extractHint` stage.
///
/// One instance per `MetalDeviceCacheEntry`; cached on the entry to keep
/// the texture cache and request objects warm across frames.
@available(macOS 14.0, *)
final class VisionHintEngine: @unchecked Sendable {

    private let cacheEntry: MetalDeviceCacheEntry
    private let textureCache: CVMetalTextureCache

    /// Vision's request objects are cheap to recreate, but holding one
    /// across frames lets Vision keep its compiled inference graph
    /// resident — saving the model-load step on every analyse frame.
    /// The lock guards the request because Vision documents
    /// `VNRequest` instances as not safe for concurrent calls.
    private let requestLock = NSLock()
    private var cachedRequest: VNGenerateForegroundInstanceMaskRequest?

    init(cacheEntry: MetalDeviceCacheEntry) throws {
        self.cacheEntry = cacheEntry
        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            nil,
            cacheEntry.device,
            nil,
            &cache
        )
        guard status == kCVReturnSuccess, let cache else {
            throw VisionHintError.textureCacheCreationFailed(status)
        }
        self.textureCache = cache
    }

    /// Reusable request that Vision can reuse across frames. Recreated
    /// lazily in case Vision invalidates it after a permanent failure.
    private func borrowRequest() -> VNGenerateForegroundInstanceMaskRequest {
        requestLock.lock()
        defer { requestLock.unlock() }
        if let cached = cachedRequest {
            return cached
        }
        let request = VNGenerateForegroundInstanceMaskRequest()
        cachedRequest = request
        return request
    }

    /// Runs Vision's foreground-instance detector on `source` and returns
    /// a Metal texture containing the union of every detected subject's
    /// scaled mask. Returns `nil` when Vision detected no foreground —
    /// the caller should fall back to `RenderStages.generateGreenHint`
    /// in that case.
    ///
    /// `source` may be in any pixel format Core Image can interpret;
    /// `.rgba16Float` and `.rgba8Unorm` both work. The returned texture is
    /// `.r8Unorm` at Vision's preferred resolution (typically the input
    /// dimensions but Vision may scale internally). Callers feed it into
    /// `RenderStages.extractHint` with `layout=1` to resample to the
    /// pre-inference target dimensions.
    func generateMask(source: any MTLTexture) throws -> VisionMask? {
        guard let baseImage = CIImage(mtlTexture: source, options: nil) else {
            PluginLog.notice("Vision hint: CIImage(mtlTexture:) returned nil for source format \(source.pixelFormat.rawValue).")
            return nil
        }
        // `CIImage(mtlTexture:)` yields a bottom-left origin image.
        // Pass `.downMirrored` as the `orientation` parameter so
        // Vision interprets the bytes correctly — pre-orienting the
        // CIImage with `.oriented(...)` was producing inconsistent
        // results because Core Image and Vision interpret the
        // orientation hint differently.
        let handler = VNImageRequestHandler(
            ciImage: baseImage,
            orientation: .downMirrored,
            options: [:]
        )
        let request = borrowRequest()

        do {
            try handler.perform([request])
        } catch {
            PluginLog.error("Vision hint perform failed: \(error.localizedDescription)")
            throw VisionHintError.requestFailed(error)
        }

        guard let observation = request.results?.first else {
            PluginLog.notice("Vision hint: no observation returned from foreground request.")
            return nil
        }
        guard !observation.allInstances.isEmpty else {
            PluginLog.notice("Vision hint: observation has zero instances — falling back to green-bias hint.")
            return nil
        }

        let maskBuffer: CVPixelBuffer
        do {
            maskBuffer = try observation.generateScaledMaskForImage(
                forInstances: observation.allInstances,
                from: handler
            )
        } catch {
            PluginLog.error("Vision hint mask scaling failed: \(error.localizedDescription)")
            throw VisionHintError.maskGenerationFailed(error)
        }
        PluginLog.notice("Vision hint: produced \(observation.allInstances.count) instance mask(s) at \(CVPixelBufferGetWidth(maskBuffer))×\(CVPixelBufferGetHeight(maskBuffer)).")
        return try wrapAsMetalTexture(pixelBuffer: maskBuffer)
    }

    /// Wraps a Vision mask CVPixelBuffer as an `.r8Unorm` MTLTexture via
    /// `CVMetalTextureCache`. Apple Silicon's unified memory means this
    /// is a pointer alias, not a copy.
    private func wrapAsMetalTexture(pixelBuffer: CVPixelBuffer) throws -> VisionMask {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .r8Unorm,
            width,
            height,
            0,
            &cvTexture
        )
        guard status == kCVReturnSuccess,
              let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture)
        else {
            throw VisionHintError.textureWrappingFailed(status)
        }
        return VisionMask(
            texture: texture,
            cvTexture: cvTexture,
            pixelBuffer: pixelBuffer
        )
    }

    /// Drops any cached request state. Called when the cache entry is
    /// torn down so Vision releases its compiled inference graph.
    func releaseCachedResources() {
        requestLock.lock()
        cachedRequest = nil
        requestLock.unlock()
        CVMetalTextureCacheFlush(textureCache, 0)
    }
}
