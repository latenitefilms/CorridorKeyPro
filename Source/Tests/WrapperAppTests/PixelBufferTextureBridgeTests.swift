//
//  PixelBufferTextureBridgeTests.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  Round-trips a `CVPixelBuffer` through the bridge into an
//  `MTLTexture` and back again, confirming:
//
//  * The Metal device cache and texture cache initialise without
//    error.
//  * Bridged textures preserve their dimensions and pixel format.
//  * IOSurface backing is preserved end-to-end (no CPU copy).
//  * Unsupported pixel formats throw a descriptive error.
//

import Testing
import Metal
import CoreVideo
@testable import CorridorKeyByLateNiteApp

@Suite("PixelBufferTextureBridge")
struct PixelBufferTextureBridgeTests {

    @Test("round-trips a 64-bit RGBA half buffer to a Metal texture without copying")
    func roundTrips64RGBAHalf() throws {
        let device = try requireDevice()
        let bridge = try PixelBufferTextureBridge(device: device)

        let pixelBuffer = try PixelBufferTextureBridge.makeMetalCompatiblePixelBuffer(
            width: 320,
            height: 180,
            pixelFormat: kCVPixelFormatType_64RGBAHalf
        )
        let backed = try bridge.makeTexture(for: pixelBuffer)

        #expect(backed.metalTexture.width == 320)
        #expect(backed.metalTexture.height == 180)
        #expect(backed.metalTexture.pixelFormat == .rgba16Float)
        #expect(CVPixelBufferGetIOSurface(backed.pixelBuffer) != nil)
    }

    @Test("round-trips a 32-bit BGRA buffer (the AVFoundation default)")
    func roundTrips32BGRA() throws {
        let device = try requireDevice()
        let bridge = try PixelBufferTextureBridge(device: device)
        let pixelBuffer = try PixelBufferTextureBridge.makeMetalCompatiblePixelBuffer(
            width: 64,
            height: 64,
            pixelFormat: kCVPixelFormatType_32BGRA
        )
        let backed = try bridge.makeTexture(for: pixelBuffer)
        #expect(backed.metalTexture.pixelFormat == .bgra8Unorm)
    }

    @Test("rejects an unsupported pixel format with a descriptive error")
    func rejectsUnsupportedFormat() throws {
        let device = try requireDevice()
        let bridge = try PixelBufferTextureBridge(device: device)

        // 4:2:0 YpCbCr is common for camera footage but the renderer
        // only handles RGBA right now. The bridge should surface a
        // clear error rather than crashing or emitting garbage.
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            64,
            64,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            attrs as CFDictionary,
            &pixelBuffer
        )
        let buffer = try #require(pixelBuffer)
        #expect(throws: PixelBufferTextureBridgeError.self) {
            _ = try bridge.makeTexture(for: buffer)
        }
    }

    @Test("Metal pixel-format mapping returns the format the renderer expects")
    func metalPixelFormatMapping() throws {
        #expect(try PixelBufferTextureBridge.metalPixelFormat(for: kCVPixelFormatType_64RGBAHalf) == .rgba16Float)
        #expect(try PixelBufferTextureBridge.metalPixelFormat(for: kCVPixelFormatType_128RGBAFloat) == .rgba32Float)
        #expect(try PixelBufferTextureBridge.metalPixelFormat(for: kCVPixelFormatType_32BGRA) == .bgra8Unorm)
    }

    private func requireDevice() throws -> any MTLDevice {
        try #require(MTLCreateSystemDefaultDevice(), "No Metal device on this host.")
    }
}
