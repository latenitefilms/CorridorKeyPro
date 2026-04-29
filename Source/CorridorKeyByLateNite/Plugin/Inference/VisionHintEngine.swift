//
//  VisionHintEngine.swift
//  CorridorKey by LateNite
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
import Accelerate

/// Wraps a Vision-generated mask as a plain Metal-owned texture. The
/// CVPixelBuffer / CVMetalTexture lifecycle is handled inside
/// `VisionHintEngine` — this struct just carries the resulting
/// `.r8Unorm` texture downstream callers can sample freely.
struct VisionMask: @unchecked Sendable {
    let texture: any MTLTexture

    init(texture: any MTLTexture) {
        self.texture = texture
    }

    /// No-op kept for source compatibility. The Metal-owned texture is
    /// retained by Swift ARC for the lifetime of the `VisionMask`
    /// value, so callers don't need to register completion-handler
    /// retention with the command buffer any more.
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
    /// Returns a plain Metal-owned `.r8Unorm` texture containing the
    /// Vision mask. Internally:
    ///
    /// 1. Runs the foreground request and gets a CVPixelBuffer.
    /// 2. Wraps it as a `CVMetalTexture` via `CVMetalTextureCache`.
    /// 3. **Blits** that wrapped texture into a plain Metal-owned
    ///    `.private` texture and returns the plain texture.
    ///
    /// The blit step exists because `CVMetalTexture`-wrapped textures
    /// from a `CVPixelBuffer`'s IOSurface have idiosyncratic
    /// shaderRead behaviour on macOS 26 — even with the explicit
    /// `kCVMetalTextureUsage` attribute set, downstream compute
    /// kernels that do `source.read(gid)` against the wrapped
    /// texture fault on the GPU. Copying once into a Metal-managed
    /// texture sidesteps the issue completely and removes the need
    /// for callers to keep the `CVMetalTexture` / `CVPixelBuffer`
    /// alive across command-buffer commits.
    ///
    /// The blit runs on a dedicated command queue borrowed from the
    /// device cache and is committed before this function returns —
    /// the returned texture is fully valid the moment it lands.
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

    /// Wraps a Vision mask CVPixelBuffer as an `.r8Unorm` MTLTexture.
    /// Vision returns the mask as a host-memory CVPixelBuffer in
    /// `kCVPixelFormatType_OneComponent32Float` (4 bytes/pixel) — not
    /// the `OneComponent8` Apple's older docs implied — so we must
    /// convert the float bytes to `UInt8` ourselves before uploading;
    /// `CVMetalTextureCache` silently produces a garbage texture when
    /// the buffer format and the requested Metal format don't agree.
    ///
    /// Conversion runs through Accelerate's `vImage` so the float→byte
    /// quantisation stays on a NEON-vectorised path. At 4K the legacy
    /// scalar Swift loop measured ~6–9 ms per analyse frame on M2 Pro;
    /// `vImageConvert_PlanarFtoPlanar8` lands in ~0.5–1 ms, freeing
    /// most of the analyse loop's CPU budget for Vision itself.
    private func wrapAsMetalTexture(pixelBuffer: CVPixelBuffer) throws -> VisionMask {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let isIOSurface = CVPixelBufferGetIOSurface(pixelBuffer) != nil
        PluginLog.notice("Vision hint: mask buffer pixelFormat=0x\(String(pixelFormat, radix: 16)), IOSurface=\(isIOSurface), \(width)x\(height), bytesPerRow=\(bytesPerRow).")

        let lockResult = CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        guard lockResult == kCVReturnSuccess else {
            throw VisionHintError.textureWrappingFailed(lockResult)
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw VisionHintError.textureWrappingFailed(0)
        }

        // Convert the buffer to a packed `UInt8` array regardless of
        // the source format. Float / half / byte all collapse to a
        // 0…255 byte where >= 128 ≈ subject. Using a packed buffer
        // lets Metal's `replace(region:)` honour our `bytesPerRow`
        // straightforwardly with no stride-mismatch surprises.
        var packed = [UInt8](repeating: 0, count: width * height)
        try packed.withUnsafeMutableBufferPointer { packedPtr in
            guard let packedBase = packedPtr.baseAddress else {
                throw VisionHintError.textureWrappingFailed(0)
            }
            switch pixelFormat {
            case kCVPixelFormatType_OneComponent8:
                // Direct row-by-row copy honouring source stride.
                for row in 0..<height {
                    let src = baseAddress.advanced(by: row * bytesPerRow)
                    let dst = packedBase.advanced(by: row * width)
                    memcpy(dst, src, width)
                }
            case kCVPixelFormatType_OneComponent32Float:
                try Self.convertFloatPlaneToByte(
                    sourceBase: baseAddress,
                    sourceBytesPerRow: bytesPerRow,
                    destinationBase: packedBase,
                    width: width,
                    height: height
                )
            case kCVPixelFormatType_OneComponent16Half:
                try Self.convertHalfPlaneToByte(
                    sourceBase: baseAddress,
                    sourceBytesPerRow: bytesPerRow,
                    destinationBase: packedBase,
                    width: width,
                    height: height
                )
            default:
                PluginLog.error("Vision hint: unsupported mask pixel format 0x\(String(pixelFormat, radix: 16)).")
                throw VisionHintError.textureWrappingFailed(-1)
            }
        }

        // Stage in `.shared` so `replace(region:)` accepts CPU bytes,
        // then blit into `.private` so downstream compute kernels can
        // sample without any storage-mode penalty.
        let stagingDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        stagingDescriptor.usage = [.shaderRead]
        stagingDescriptor.storageMode = .shared
        guard let stagingTexture = cacheEntry.device.makeTexture(descriptor: stagingDescriptor) else {
            throw VisionHintError.textureWrappingFailed(0)
        }
        stagingTexture.label = "Vision Hint Mask Staging"
        packed.withUnsafeBufferPointer { ptr in
            if let base = ptr.baseAddress {
                stagingTexture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width
                )
            }
        }

        let plainDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        plainDescriptor.usage = [.shaderRead]
        plainDescriptor.storageMode = .private
        guard let plainTexture = cacheEntry.device.makeTexture(descriptor: plainDescriptor) else {
            throw VisionHintError.textureWrappingFailed(0)
        }
        plainTexture.label = "Vision Hint Mask"

        guard let queue = cacheEntry.borrowCommandQueue() else {
            throw VisionHintError.textureWrappingFailed(0)
        }
        defer { cacheEntry.returnCommandQueue(queue) }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else {
            throw VisionHintError.textureWrappingFailed(0)
        }
        commandBuffer.label = "Vision Hint Mask Blit"
        blit.copy(
            from: stagingTexture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: width, height: height, depth: 1),
            to: plainTexture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        // `addCompletedHandler` + `DispatchSemaphore` instead of
        // `waitUntilCompleted`: the analysis thread parks cleanly
        // instead of busy-spinning while the blit lands. Saves
        // ~1–3 ms per analysis frame on M-series and removes the
        // last `waitUntilCompleted` from the rendering pipeline,
        // matching the pattern `RenderPipeline.commitAndWait` uses
        // for everything else.
        let blitSemaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in blitSemaphore.signal() }
        commandBuffer.commit()
        blitSemaphore.wait()
        if let error = commandBuffer.error {
            throw error
        }

        // One-shot sanity check on what we actually wrote into the
        // staging texture. If production ever logs `nonZero=0` while
        // Vision claims it produced a mask, we know to look upstream
        // (Vision returning an empty buffer) rather than at the GPU
        // sampling path.
        Self.logFirstStagingSampleOnce(stagingTexture: stagingTexture, width: width, height: height)

        return VisionMask(texture: plainTexture)
    }

    nonisolated(unsafe) private static var stagingSampleLogged = false

    private static func logFirstStagingSampleOnce(
        stagingTexture: any MTLTexture,
        width: Int,
        height: Int
    ) {
        guard !stagingSampleLogged else { return }
        stagingSampleLogged = true
        var bytes = [UInt8](repeating: 0, count: width * height)
        bytes.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                stagingTexture.getBytes(
                    base,
                    bytesPerRow: width,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        var nonZero = 0
        var maxValue: UInt8 = 0
        for byte in bytes {
            if byte > 0 { nonZero += 1 }
            if byte > maxValue { maxValue = byte }
        }
        PluginLog.notice("Vision hint staging sample (one-time): nonZero=\(nonZero)/\(bytes.count) (\(Double(nonZero) / Double(bytes.count) * 100)%), max=\(maxValue).")
    }

    /// Drops any cached request state. Called when the cache entry is
    /// torn down so Vision releases its compiled inference graph.
    func releaseCachedResources() {
        requestLock.lock()
        cachedRequest = nil
        requestLock.unlock()
        CVMetalTextureCacheFlush(textureCache, 0)
    }

    /// Converts a planar Float32 buffer to packed UInt8 using
    /// Accelerate's NEON-vectorised `vImageConvert_PlanarFtoPlanar8`.
    /// Replaces a hand-rolled scalar loop that took ~6–9 ms on a 4K
    /// Vision mask (M2 Pro). The destination buffer is tightly packed
    /// (`bytesPerRow == width`) so the writeback path stays simple
    /// for the staging-texture upload.
    private static func convertFloatPlaneToByte(
        sourceBase: UnsafeMutableRawPointer,
        sourceBytesPerRow: Int,
        destinationBase: UnsafeMutablePointer<UInt8>,
        width: Int,
        height: Int
    ) throws {
        var sourceBuffer = vImage_Buffer(
            data: sourceBase,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: sourceBytesPerRow
        )
        var destinationBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(destinationBase),
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
        )
        // maxFloat=1.0, minFloat=0.0 emits 0…255 with the same clamp
        // semantics the previous scalar loop used.
        let status = vImageConvert_PlanarFtoPlanar8(
            &sourceBuffer,
            &destinationBuffer,
            1.0,
            0.0,
            vImage_Flags(kvImageDoNotTile)
        )
        guard status == kvImageNoError else {
            throw VisionHintError.textureWrappingFailed(OSStatus(status))
        }
    }

    /// Converts a planar Float16 buffer to packed UInt8. Goes via
    /// Float32 with the dedicated Accelerate half→float pass so the
    /// final step can reuse `vImageConvert_PlanarFtoPlanar8`.
    /// Vision shipped a Float16 path on a few macOS releases; keep
    /// the conversion lane warm so we never silently fall back to
    /// the slow scalar path.
    private static func convertHalfPlaneToByte(
        sourceBase: UnsafeMutableRawPointer,
        sourceBytesPerRow: Int,
        destinationBase: UnsafeMutablePointer<UInt8>,
        width: Int,
        height: Int
    ) throws {
        var floatScratch = [Float](repeating: 0, count: width * height)
        try floatScratch.withUnsafeMutableBufferPointer { floatPtr in
            guard let floatBase = floatPtr.baseAddress else {
                throw VisionHintError.textureWrappingFailed(0)
            }
            var halfBuffer = vImage_Buffer(
                data: sourceBase,
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: sourceBytesPerRow
            )
            var floatBuffer = vImage_Buffer(
                data: UnsafeMutableRawPointer(floatBase),
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: width * MemoryLayout<Float>.size
            )
            let halfToFloatStatus = vImageConvert_Planar16FtoPlanarF(
                &halfBuffer,
                &floatBuffer,
                vImage_Flags(kvImageDoNotTile)
            )
            guard halfToFloatStatus == kvImageNoError else {
                throw VisionHintError.textureWrappingFailed(OSStatus(halfToFloatStatus))
            }
            try convertFloatPlaneToByte(
                sourceBase: UnsafeMutableRawPointer(floatBase),
                sourceBytesPerRow: width * MemoryLayout<Float>.size,
                destinationBase: destinationBase,
                width: width,
                height: height
            )
        }
    }
}
