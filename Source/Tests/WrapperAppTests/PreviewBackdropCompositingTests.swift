//
//  PreviewBackdropCompositingTests.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  Regression tests for the bug where the keyed image's
//  premultiplied-black background got drawn opaque over the
//  user's chosen preview backdrop.
//
//  The render pipeline emits Processed-mode frames as
//  `(foreground * alpha, alpha)` — premultiplied alpha. The preview
//  view has to composite that output onto a checkerboard / colour
//  backdrop using premultiplied source-over blending; if it doesn't,
//  every transparent pixel reads as solid black no matter what the
//  user picks for the backdrop. This file synthesises exactly that
//  scenario (a premultiplied texture with one opaque red pixel and
//  one fully-transparent pixel) and reads back the composited
//  drawable to confirm the backdrop is visible.
//

import Testing
import Foundation
import Metal
import CoreVideo
@testable import CorridorKeyToolboxApp

@Suite("Preview backdrop compositing", .serialized)
struct PreviewBackdropCompositingTests {

    @Test("premultiplied source with alpha=0 reveals the chosen backdrop instead of being baked to black")
    @MainActor
    func transparentRegionShowsBackdrop() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let backdropTester = try BackdropPreviewTester(device: device)

        // 2-pixel premultiplied source: column 0 is fully transparent
        // (RGBA = 0, 0, 0, 0), column 1 is opaque red (RGBA = 1, 0, 0, 1).
        // After composition over the red backdrop:
        //   * column 0: backdrop should show through ⇒ red.
        //   * column 1: source's red ⇒ red.
        // Both columns reading red proves the transparent area is
        // not being baked to black.
        let source = try Self.makePremultipliedTexture(
            device: device,
            pixels: [(r: 0, g: 0, b: 0, a: 0), (r: 1, g: 0, b: 0, a: 1)]
        )
        let result = try backdropTester.composite(source: source, backdrop: .red)

        let transparentRegion = result.sample(x: 0, y: 0)
        let opaqueRegion = result.sample(x: 1, y: 0)

        #expect(transparentRegion.r > 0.5,
                "Transparent area should show the red backdrop, instead got \(transparentRegion).")
        #expect(opaqueRegion.r > 0.5,
                "Opaque-red source should still render red, instead got \(opaqueRegion).")
        // The backdrop must NOT be black for the transparent region.
        // This is the assertion that would have caught the bug:
        // the previous shader returned `float4(rgb, 1.0)` and
        // forced source-over blending to ignore the matte alpha.
        #expect(transparentRegion.r > 0.05 || transparentRegion.g > 0.05 || transparentRegion.b > 0.05,
                "Transparent area is rendering pure black — the keyed background is being baked over the backdrop.")
    }

    @Test("premultiplied half-transparent source blends 50/50 with the backdrop")
    @MainActor
    func halfTransparentRegionMixesWithBackdrop() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let backdropTester = try BackdropPreviewTester(device: device)

        // Premultiplied: rgb is foreground colour scaled by alpha.
        // Pure white at 50% alpha ⇒ stored as (0.5, 0.5, 0.5, 0.5).
        // Composite over yellow backdrop should yield ~50/50 mix.
        let source = try Self.makePremultipliedTexture(
            device: device,
            pixels: [(r: 0.5, g: 0.5, b: 0.5, a: 0.5)]
        )
        let result = try backdropTester.composite(source: source, backdrop: .yellow)
        let pixel = result.sample(x: 0, y: 0)
        // With premultiplied source-over: final.r = source.r + dest.r * (1 - source.a)
        //   = 0.5 + 1.0 * 0.5 = 1.0. green: 0.5 + 0.85 * 0.5 ≈ 0.93. blue: 0.5 + 0 * 0.5 = 0.5.
        // BGRA8Unorm stores 8-bit values, so each channel rounds.
        #expect(abs(pixel.r - 1.0) < 0.05, "Red channel \(pixel.r) is not the expected ~1.0 (foreground white + half-yellow).")
        #expect(pixel.g > 0.7 && pixel.g < 1.0, "Green channel \(pixel.g) is not within the expected ~0.93 band.")
        #expect(pixel.b > 0.4 && pixel.b < 0.6, "Blue channel \(pixel.b) is not the expected ~0.5 (foreground only, backdrop yellow has no blue).")
    }

    @Test("opaque source replaces the backdrop entirely")
    @MainActor
    func opaqueSourceCoversBackdrop() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let backdropTester = try BackdropPreviewTester(device: device)
        let source = try Self.makePremultipliedTexture(
            device: device,
            pixels: [(r: 0.10, g: 0.40, b: 0.95, a: 1.0)]
        )
        let result = try backdropTester.composite(source: source, backdrop: .yellow)
        let pixel = result.sample(x: 0, y: 0)
        // Opaque source with alpha=1 should fully cover the
        // backdrop — yellow should not bleed through at all.
        #expect(abs(pixel.r - 0.10) < 0.05, "Red channel \(pixel.r) — opaque source should not be tinted yellow.")
        #expect(abs(pixel.g - 0.40) < 0.05, "Green channel \(pixel.g) — opaque source should not be tinted yellow.")
        #expect(abs(pixel.b - 0.95) < 0.05, "Blue channel \(pixel.b) — opaque source should not be tinted yellow.")
    }

    @Test("backdrop choice is reflected in the rendered transparent area")
    @MainActor
    func backdropChoicePropagates() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let backdropTester = try BackdropPreviewTester(device: device)
        let transparentSource = try Self.makePremultipliedTexture(
            device: device,
            pixels: [(r: 0, g: 0, b: 0, a: 0)]
        )

        // White backdrop should produce a near-white pixel.
        let onWhite = try backdropTester.composite(source: transparentSource, backdrop: .white)
        let whitePixel = onWhite.sample(x: 0, y: 0)
        #expect(whitePixel.r > 0.9 && whitePixel.g > 0.9 && whitePixel.b > 0.9,
                "Transparent source on white backdrop should render white, got \(whitePixel).")

        // Black backdrop should produce a near-black pixel.
        let onBlack = try backdropTester.composite(source: transparentSource, backdrop: .black)
        let blackPixel = onBlack.sample(x: 0, y: 0)
        #expect(blackPixel.r < 0.1 && blackPixel.g < 0.1 && blackPixel.b < 0.1,
                "Transparent source on black backdrop should render black, got \(blackPixel).")
    }

    // MARK: - Helpers

    /// Builds a 1-row premultiplied RGBA16Float texture from a list of
    /// `(r, g, b, a)` tuples. Each tuple is interpreted as the
    /// premultiplied compose-shader output: `rgb` is the foreground
    /// already scaled by alpha, `a` is the matte coverage. Width
    /// equals the pixel count.
    private static func makePremultipliedTexture(
        device: any MTLDevice,
        pixels: [(r: Float, g: Float, b: Float, a: Float)]
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: pixels.count,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(device.makeTexture(descriptor: descriptor))
        var halves = [UInt16](repeating: 0, count: pixels.count * 4)
        for (index, pixel) in pixels.enumerated() {
            halves[index * 4 + 0] = Float16(pixel.r).bitPattern
            halves[index * 4 + 1] = Float16(pixel.g).bitPattern
            halves[index * 4 + 2] = Float16(pixel.b).bitPattern
            halves[index * 4 + 3] = Float16(pixel.a).bitPattern
        }
        let bytesPerRow = pixels.count * 4 * MemoryLayout<UInt16>.size
        halves.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, pixels.count, 1),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: bytesPerRow
                )
            }
        }
        return texture
    }
}

/// Owns the same `MTKView`-based pipeline `MetalPreviewView` uses
/// (textured-quad render with premultiplied source-over blending +
/// chequerboard / solid-colour backdrops) so the unit tests can
/// drive it without standing up a real window. Each `composite(...)`
/// call renders one frame into an off-screen `bgra8Unorm` texture
/// the test reads back to assert the per-pixel result.
@MainActor
final class BackdropPreviewTester {
    let device: any MTLDevice
    private let coordinator: MetalPreviewView.Coordinator

    init(device: any MTLDevice) throws {
        self.device = device
        self.coordinator = MetalPreviewView.Coordinator(device: device)
    }

    func composite(source: any MTLTexture, backdrop: PreviewBackdrop) throws -> CompositedSurface {
        let outputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: source.width,
            height: max(source.height, 1),
            mipmapped: false
        )
        outputDescriptor.usage = [.renderTarget, .shaderRead]
        outputDescriptor.storageMode = .shared
        let outputTexture = try #require(device.makeTexture(descriptor: outputDescriptor))

        coordinator.update(
            sourceTexture: source,
            aspectFitSize: CGSize(width: source.width, height: source.height),
            backdrop: backdrop,
            previewFrame: nil
        )
        try coordinator.renderForTesting(into: outputTexture)
        return try CompositedSurface(reading: outputTexture)
    }
}

/// Convenience wrapper around the read-back BGRA8 bytes so each
/// test can ask for a single sample by `(x, y)`.
struct CompositedSurface {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init(reading texture: any MTLTexture) throws {
        let width = texture.width
        let height = texture.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes { rawBytes in
            if let base = rawBytes.baseAddress {
                texture.getBytes(
                    base,
                    bytesPerRow: width * 4,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        self.width = width
        self.height = height
        self.pixels = bytes
    }

    func sample(x: Int, y: Int) -> (r: Float, g: Float, b: Float, a: Float) {
        let offset = (y * width + x) * 4
        // BGRA8Unorm: byte 0 = blue, byte 1 = green, byte 2 = red, byte 3 = alpha.
        return (
            r: Float(pixels[offset + 2]) / 255,
            g: Float(pixels[offset + 1]) / 255,
            b: Float(pixels[offset + 0]) / 255,
            a: Float(pixels[offset + 3]) / 255
        )
    }
}
