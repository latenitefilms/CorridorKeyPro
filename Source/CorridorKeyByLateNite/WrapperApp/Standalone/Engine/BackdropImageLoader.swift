//
//  BackdropImageLoader.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Loads a user-imported still image into a `MTLTexture` the
//  preview surface can sample as its custom backdrop. Decoding +
//  texture upload happens off the main actor so the import sheet
//  doesn't stall the editor on large source files.
//
//  We deliberately load the image as a non-sRGB texture: the
//  preview pipeline writes to a `bgra8Unorm` framebuffer (no
//  colour-management round-trip), and the existing checker /
//  solid-colour backdrops also pass their bytes straight to the
//  framebuffer. Marking the texture sRGB here would gamma-encode
//  twice and the imported image would look washed out compared
//  to how Photos.app renders it on the same monitor.
//

import Foundation
import Metal
import MetalKit
import CoreGraphics
import ImageIO

enum BackdropImageLoaderError: Error, LocalizedError {
    case unsupportedFormat
    case textureCreationFailed(underlying: any Error)
    case fileTooLarge(bytes: Int)

    var errorDescription: String? {
        switch self {
        case .unsupportedFormat:
            return "Couldn't decode the image. Try exporting it as PNG or JPEG."
        case .textureCreationFailed(let underlying):
            return "Couldn't load the image into Metal: \(underlying.localizedDescription)"
        case .fileTooLarge(let bytes):
            let mb = Double(bytes) / (1024 * 1024)
            return "Image is \(mb.formatted(.number.precision(.fractionLength(0)))) MB — pick something smaller (≤ 64 MB)."
        }
    }
}

enum BackdropImageLoader {
    /// Hard cap on input size — a 64 MB JPEG decodes to ~250 MB of
    /// uncompressed pixels at 4K, which is plenty of headroom for a
    /// preview backdrop. Anything larger is almost certainly user
    /// error (photo from a 100MP body, RAW capture, etc.) and we'd
    /// rather refuse than spend half a second on a failed import.
    static let maximumFileSize = 64 * 1024 * 1024

    /// Synchronously decodes the supplied image bytes into a
    /// `MTLTexture`. Safe to call from a `Task.detached` so the
    /// view model can keep the import off the main actor.
    static func makeTexture(from data: Data, device: any MTLDevice) throws -> any MTLTexture {
        guard data.count <= maximumFileSize else {
            throw BackdropImageLoaderError.fileTooLarge(bytes: data.count)
        }
        let loader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: NSNumber(value: false),
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue),
            .generateMipmaps: NSNumber(value: false),
            .origin: MTKTextureLoader.Origin.topLeft.rawValue
        ]
        do {
            return try loader.newTexture(data: data, options: options)
        } catch {
            throw BackdropImageLoaderError.textureCreationFailed(underlying: error)
        }
    }
}
