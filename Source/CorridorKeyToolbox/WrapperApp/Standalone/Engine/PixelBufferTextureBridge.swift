//
//  PixelBufferTextureBridge.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Bridges between AVFoundation's `CVPixelBuffer` (used by `AVAssetReader`
//  / `AVAssetWriter`) and Metal's `MTLTexture` (consumed by the Corridor
//  Key render pipeline). All frames are routed through Metal-compatible
//  IOSurface-backed pixel buffers so the hand-off is a pointer copy on
//  Apple Silicon's unified memory rather than a CPU blit.
//
//  We deliberately bridge via `device.makeTexture(descriptor:iosurface:plane:)`
//  instead of `CVMetalTextureCacheCreateTextureFromImage`. The cache
//  variant produces textures with only `.shaderRead` usage — fine for
//  the source side, but useless for the destination, which the
//  renderer needs to bind as a colour attachment in `compose(...)`.
//  Using the IOSurface API directly lets each call site declare the
//  exact usage flags it needs.
//
//  Thread safety: the bridge holds no mutable state — every API call
//  is `pure` once `init` returns — so it is trivially `@unchecked
//  Sendable`. Each `makeTexture` call independently asks the
//  `MTLDevice` for a fresh texture; there is no cache to invalidate.
//

import Foundation
import AVFoundation
import CoreVideo
import IOSurface
import Metal

/// Errors raised by the bridge. The standalone editor surfaces these as
/// alerts so the user can see why a frame failed to load or write.
enum PixelBufferTextureBridgeError: Error, CustomStringConvertible {
    case unsupportedPixelFormat(OSType)
    case textureCreationFailed
    case noIOSurface
    case pixelBufferAllocationFailed(CVReturn)

    var description: String {
        switch self {
        case .unsupportedPixelFormat(let osType):
            return "CorridorKey by LateNite cannot bridge pixel format 0x\(String(osType, radix: 16))."
        case .textureCreationFailed:
            return "CorridorKey by LateNite could not wrap a pixel buffer as a Metal texture."
        case .noIOSurface:
            return "CorridorKey by LateNite received a CVPixelBuffer without an IOSurface; the AVFoundation reader was not configured for Metal compatibility."
        case .pixelBufferAllocationFailed(let status):
            return "CorridorKey by LateNite could not allocate a destination pixel buffer (status \(status))."
        }
    }
}

/// Bridges `CVPixelBuffer`s into `MTLTexture`s with caller-controlled
/// usage flags. One instance is held per `MTLDevice` so multi-GPU
/// systems do not contend on a single bridge.
final class PixelBufferTextureBridge: @unchecked Sendable {
    let device: any MTLDevice

    init(device: any MTLDevice) throws {
        self.device = device
    }

    /// No-op kept for API stability; the IOSurface bridge has no
    /// internal cache to flush.
    func flushUnused() { /* no-op */ }

    /// Wraps `pixelBuffer` as an `MTLTexture` without copying the pixel
    /// data. The returned texture aliases the IOSurface backing the
    /// pixel buffer; keep the pixel buffer alive for the lifetime of
    /// any GPU work that touches it.
    ///
    /// `usage` must include every usage flag the caller will rely on.
    /// Pass `.shaderRead` for source textures, `[.shaderRead,
    /// .renderTarget]` for textures the render pipeline writes into
    /// via a render pass, and `[.shaderRead, .shaderWrite]` for
    /// textures a compute kernel writes into.
    func makeTexture(
        for pixelBuffer: CVPixelBuffer,
        usage: MTLTextureUsage = .shaderRead
    ) throws -> MetalBackedPixelBuffer {
        let pixelFormatType = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let metalPixelFormat = try Self.metalPixelFormat(for: pixelFormatType)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        guard let ioSurface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else {
            throw PixelBufferTextureBridgeError.noIOSurface
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: metalPixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor, iosurface: ioSurface, plane: 0) else {
            throw PixelBufferTextureBridgeError.textureCreationFailed
        }
        return MetalBackedPixelBuffer(pixelBuffer: pixelBuffer, metalTexture: texture)
    }

    // MARK: - Pixel buffer factories

    /// Creates an IOSurface-backed pixel buffer suitable for both AV
    /// Foundation and Metal. Used as the destination for renders that
    /// will be encoded to ProRes by `AVAssetWriter`.
    static func makeMetalCompatiblePixelBuffer(
        width: Int,
        height: Int,
        pixelFormat: OSType
    ) throws -> CVPixelBuffer {
        let attributes: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormat,
            attributes as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = buffer else {
            throw PixelBufferTextureBridgeError.pixelBufferAllocationFailed(status)
        }
        return pixelBuffer
    }

    /// Maps an AV Foundation pixel format to the Metal equivalent the
    /// renderer expects. Throws on formats that the pipeline does not
    /// support so callers fall back to a known-good format instead of
    /// silently producing wrong colours.
    static func metalPixelFormat(for pixelFormat: OSType) throws -> MTLPixelFormat {
        switch pixelFormat {
        case kCVPixelFormatType_64RGBAHalf: return .rgba16Float
        case kCVPixelFormatType_128RGBAFloat: return .rgba32Float
        case kCVPixelFormatType_32BGRA: return .bgra8Unorm
        default: throw PixelBufferTextureBridgeError.unsupportedPixelFormat(pixelFormat)
        }
    }
}

/// Small bundle that pairs a `CVPixelBuffer` with the `MTLTexture` view
/// onto its IOSurface. Holding this struct keeps the IOSurface alive
/// (via the pixel buffer's retain) for the lifetime of any GPU work
/// touching the texture.
struct MetalBackedPixelBuffer: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let metalTexture: any MTLTexture
}
