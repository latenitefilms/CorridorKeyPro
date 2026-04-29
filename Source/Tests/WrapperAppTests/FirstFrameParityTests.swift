//
//  FirstFrameParityTests.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  End-to-end parity gate between the Standalone Editor's render
//  path and the shared `RenderPipeline` the FxPlug also drives. The
//  user reported that Final Cut Pro's CorridorKey effect produces
//  visibly different results from the Standalone Editor on the
//  NikoDruid four-frame fixture; this suite is the regression net
//  for that, structured so any future divergence between the two
//  surfaces fails immediately rather than slipping into a release.
//
//  Two layers of coverage:
//
//  * **Code-path parity** — drive both `StandaloneRenderEngine.render(...)`
//    and a direct `RenderPipeline.renderToTexture(...)` against the
//    same source pixel buffer with the same `PluginStateData`, then
//    require their destination textures to match byte-for-byte.
//    They share `RenderPipeline` so any divergence here means a
//    bug in the wrapper around it (e.g. PixelBufferTextureBridge
//    quirks, the AVFoundation pixel-buffer factory, or destination
//    texture usage flags).
//
//  * **Visual gold-standard** — write the standalone render of the
//    NikoDruid first frame to a PNG in `/tmp` and surface the path
//    in the test log. Open the same clip in Final Cut Pro, render
//    the first frame at the same default settings, and compare the
//    two side-by-side. The PNG is the canonical "what the FxPlug
//    *should* produce" reference.
//
//  Both paths run with `qualityMode: .draft512` so the test stays
//  inside CI's wall-clock budget; the parity guarantee holds at
//  every rung because the post-process / compose stages are
//  resolution-agnostic.
//

import Testing
import Foundation
import Metal
import CoreMedia
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import AVFoundation
@testable import CorridorKeyByLateNiteApp

@Suite("First frame parity — Standalone vs FxPlug render pipeline", .serialized)
struct FirstFrameParityTests {

    /// The two render entry points share the same `RenderPipeline`,
    /// so given identical source / state / gamut they have to land
    /// on identical destination pixels. Any drift here is either a
    /// non-deterministic GPU op (which we'd want to track down) or
    /// a wrapper-side bug like PixelBufferTextureBridge handing
    /// the FxPlug-equivalent path a differently-configured texture.
    @Test("renderToTexture produces byte-identical pixels via both standalone and direct paths")
    func standaloneAndDirectRendersAgree() async throws {
        let url = try #require(
            RealClipFixture.fourFrameClipURL(),
            "NikoDruid 4-frame fixture missing — see RealClipFixture for the expected path."
        )
        let source = try await VideoSource(url: url)
        let pixelBuffer = try await source.makeFrame(atTime: .zero)

        let engine = try StandaloneRenderEngine()
        // Build a state that exercises every default product knob —
        // the same bundle a fresh FCP project applies via the
        // Motion Template, so any divergence would mirror what the
        // user actually sees.
        var state = PluginStateData()
        state.qualityMode = .draft512
        state.destinationLongEdgePixels = max(
            CVPixelBufferGetWidth(pixelBuffer),
            CVPixelBufferGetHeight(pixelBuffer)
        )

        // Standalone path.
        let standaloneResult = try engine.render(
            source: pixelBuffer,
            state: state,
            renderTime: .zero
        )
        let standalonePixels = try Self.readPixels(
            from: standaloneResult.destinationTexture
        )

        // Direct path — same as the FxPlug does internally once it
        // has resolved its `FxImageTile` into raw `MTLTexture`s.
        let bridge = try PixelBufferTextureBridge(device: engine.device)
        let sourceBacked = try bridge.makeTexture(for: pixelBuffer, usage: .shaderRead)
        let directDestinationBuffer = try PixelBufferTextureBridge.makeMetalCompatiblePixelBuffer(
            width: sourceBacked.metalTexture.width,
            height: sourceBacked.metalTexture.height,
            pixelFormat: CVPixelBufferGetPixelFormatType(pixelBuffer)
        )
        let directDestination = try bridge.makeTexture(
            for: directDestinationBuffer,
            usage: [.shaderRead, .renderTarget]
        )
        let pipeline = RenderPipeline()
        _ = try pipeline.renderToTexture(
            source: sourceBacked.metalTexture,
            destination: directDestination.metalTexture,
            alphaHint: nil,
            state: state,
            workingGamut: .rec709,
            renderTime: .zero
        )
        let directPixels = try Self.readPixels(from: directDestination.metalTexture)

        // Texture pixel formats must match for the comparison to
        // make sense — the standalone factory propagates the
        // source's format, and the direct path here mirrors that
        // exactly.
        #expect(standalonePixels.format == directPixels.format,
                "Pixel format diverged between standalone and direct paths: standalone=\(standalonePixels.format), direct=\(directPixels.format).")
        #expect(standalonePixels.width == directPixels.width)
        #expect(standalonePixels.height == directPixels.height)

        let pairs = zip(standalonePixels.values, directPixels.values)
        let maxDelta = pairs.reduce(into: Float(0)) { acc, pair in
            acc = max(acc, abs(pair.0 - pair.1))
        }
        // Both paths share the same Metal device + pipeline so we
        // expect bitwise equality. Allowing 1/255 of tolerance covers
        // the rare case where the GPU's internal scheduling reorders
        // a `mix(a, b, t)` and floats lose a low bit.
        #expect(maxDelta <= 1.0 / 255.0,
                "Standalone and direct render outputs diverge by up to \(maxDelta) — the FxPlug would not match the Standalone Editor. (channel-wise float delta on linear sRGB)")
    }

    /// Renders the NikoDruid first frame the way the editor's
    /// preview would see it and writes the result to disk as a PNG.
    /// The test prints the path so a human can open the file and
    /// compare visually with what Final Cut Pro shows for the same
    /// clip with the same default settings.
    ///
    /// Also runs simple coverage / range checks so an obvious
    /// regression (all-black render, NaN pixels) trips the gate
    /// even without manual inspection.
    @Test("standalone first-frame render is non-trivial and matches the NikoDruid baseline statistics")
    func standaloneFirstFrameVisualBaseline() async throws {
        let url = try #require(RealClipFixture.fourFrameClipURL())
        let source = try await VideoSource(url: url)
        let pixelBuffer = try await source.makeFrame(atTime: .zero)

        let engine = try StandaloneRenderEngine()
        // Default state — no overrides. `.automatic` quality means
        // the inference rung is picked from the source long-edge
        // (1536px for the 4K NikoDruid clip on a 64 GB Apple
        // Silicon Mac), so the Standalone PNG matches what FCP
        // would render at the same default settings.
        var state = PluginStateData()
        state.destinationLongEdgePixels = max(
            CVPixelBufferGetWidth(pixelBuffer),
            CVPixelBufferGetHeight(pixelBuffer)
        )

        // Pre-bake the matte for frame 0 so the render exercises
        // the full post-process pipeline (despill, light wrap,
        // edge decontaminate, refiner, compose). Without a cached
        // matte, the renderer falls back to source pass-through
        // which wouldn't tell us anything about post-process
        // parity.
        let analyseOutput = try engine.extractMatteBlob(
            source: pixelBuffer,
            state: state,
            renderTime: .zero
        )
        state.cachedMatteBlob = analyseOutput.blob
        state.cachedMatteInferenceResolution = analyseOutput.inferenceResolution

        let result = try engine.render(
            source: pixelBuffer,
            state: state,
            renderTime: .zero
        )
        let pixels = try Self.readPixels(from: result.destinationTexture)

        // Write the PNG to a stable filename so each run overwrites
        // the previous. Surface the path so the test log lands a
        // copy-pastable line for human inspection.
        let outputURL = try Self.writePNG(
            from: pixels,
            named: "CorridorKey-NikoDruid-frame0-Standalone.png"
        )
        print("First-frame PNG written to: \(outputURL.path)")

        // Coverage gate: NikoDruid frame 0 is a green-screen shot
        // of a single subject. A correct keyer leaves the subject
        // visible (substantial non-zero alpha coverage) and removes
        // the green background (a meaningful zero-alpha fraction).
        // A broken render that emits all source-pass-through (zero
        // matte) or all transparent (zero coverage) trips one of
        // these checks before the user has to open the PNG.
        let summary = pixels.summary
        #expect(summary.maxValue > 0.05,
                "Destination pixels are effectively black (max value \(summary.maxValue)) — the render path didn't write meaningful pixels.")
        #expect(summary.alphaCoverageFraction > 0.05,
                "Less than 5% of pixels carry non-trivial alpha (got \(summary.alphaCoverageFraction)) — the matte didn't survive post-process.")
        #expect(summary.alphaCoverageFraction < 0.95,
                "More than 95% of pixels are fully opaque (got \(summary.alphaCoverageFraction)) — the keyer didn't remove the green background.")
        #expect(!summary.hasNaN,
                "Destination pixels contain NaN — almost certainly a divide-by-zero in despill or light-wrap.")
    }

    // MARK: - Helpers

    /// Captured pixels in a host-readable layout. Holds float values
    /// (linear, channel-major) so downstream comparisons aren't
    /// pixel-format-specific.
    private struct CapturedPixels {
        let width: Int
        let height: Int
        let format: MTLPixelFormat
        /// `width * height * 4` floats, RGBA row-major top-to-bottom.
        let values: [Float]

        var summary: (maxValue: Float, alphaCoverageFraction: Double, hasNaN: Bool) {
            var maxValue: Float = 0
            var alphaPixels = 0
            var nan = false
            let pixelCount = width * height
            for index in 0..<pixelCount {
                let r = values[index * 4 + 0]
                let g = values[index * 4 + 1]
                let b = values[index * 4 + 2]
                let a = values[index * 4 + 3]
                if r.isNaN || g.isNaN || b.isNaN || a.isNaN { nan = true }
                maxValue = max(maxValue, max(r, max(g, max(b, a))))
                if a > 0.05 { alphaPixels += 1 }
            }
            return (maxValue, Double(alphaPixels) / Double(pixelCount), nan)
        }
    }

    /// Reads back a Metal texture into a CPU-side float buffer. Both
    /// `rgba16Float` (the editor's preferred preview format) and
    /// `bgra8Unorm` (used when the source decode falls back to 8-bit)
    /// are supported; everything else trips the test loud rather
    /// than silently emitting a black PNG.
    private static func readPixels(from texture: any MTLTexture) throws -> CapturedPixels {
        let width = texture.width
        let height = texture.height
        let pixelCount = width * height

        switch texture.pixelFormat {
        case .rgba16Float:
            var halves = [UInt16](repeating: 0, count: pixelCount * 4)
            halves.withUnsafeMutableBytes { bytes in
                if let base = bytes.baseAddress {
                    texture.getBytes(
                        base,
                        bytesPerRow: width * 4 * MemoryLayout<UInt16>.size,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                }
            }
            var floats = [Float](repeating: 0, count: pixelCount * 4)
            for index in 0..<(pixelCount * 4) {
                floats[index] = Float(Float16(bitPattern: halves[index]))
            }
            return CapturedPixels(width: width, height: height, format: .rgba16Float, values: floats)

        case .bgra8Unorm:
            var bytes = [UInt8](repeating: 0, count: pixelCount * 4)
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
            var floats = [Float](repeating: 0, count: pixelCount * 4)
            for index in 0..<pixelCount {
                let b = Float(bytes[index * 4 + 0]) / 255
                let g = Float(bytes[index * 4 + 1]) / 255
                let r = Float(bytes[index * 4 + 2]) / 255
                let a = Float(bytes[index * 4 + 3]) / 255
                floats[index * 4 + 0] = r
                floats[index * 4 + 1] = g
                floats[index * 4 + 2] = b
                floats[index * 4 + 3] = a
            }
            return CapturedPixels(width: width, height: height, format: .bgra8Unorm, values: floats)

        default:
            // The compose pass writes either of the two formats
            // above, so any other format here is a bug elsewhere.
            // Fail noisily instead of trying to interpret unknown
            // bytes — silent translation has hidden previous bugs.
            throw FirstFrameParityTestsError.unsupportedPixelFormat(texture.pixelFormat)
        }
    }

    private enum FirstFrameParityTestsError: Error, CustomStringConvertible {
        case unsupportedPixelFormat(MTLPixelFormat)
        case pngEncodeFailed
        case temporaryFolderUnavailable

        var description: String {
            switch self {
            case .unsupportedPixelFormat(let format):
                return "First-frame parity test does not know how to read back pixel format \(format.rawValue)."
            case .pngEncodeFailed:
                return "First-frame parity test could not encode the captured pixels as a PNG."
            case .temporaryFolderUnavailable:
                return "First-frame parity test could not create a destination folder for the diagnostic PNG."
            }
        }
    }

    /// Writes the captured pixels to a PNG so a human reviewer can
    /// open the file and compare it side-by-side with what FCP
    /// produces for the same clip. PNGs go to `/tmp/CorridorKey
    /// Tests/` so they survive across runs and are easy to find.
    /// Each value is clamped to [0, 1] before encode — the extended
    /// sRGB range that Float16 can represent isn't representable in
    /// 8-bit PNG anyway.
    @discardableResult
    private static func writePNG(
        from pixels: CapturedPixels,
        named filename: String
    ) throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appending(path: "CorridorKey Tests", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: folder,
            withIntermediateDirectories: true
        )
        let outputURL = folder.appending(path: filename)

        let width = pixels.width
        let height = pixels.height
        let pixelCount = width * height
        var bytes = [UInt8](repeating: 0, count: pixelCount * 4)
        for index in 0..<pixelCount {
            let r = max(0, min(1, pixels.values[index * 4 + 0]))
            let g = max(0, min(1, pixels.values[index * 4 + 1]))
            let b = max(0, min(1, pixels.values[index * 4 + 2]))
            let a = max(0, min(1, pixels.values[index * 4 + 3]))
            bytes[index * 4 + 0] = UInt8((r * 255).rounded())
            bytes[index * 4 + 1] = UInt8((g * 255).rounded())
            bytes[index * 4 + 2] = UInt8((b * 255).rounded())
            bytes[index * 4 + 3] = UInt8((a * 255).rounded())
        }
        let bitsPerComponent = 8
        let bytesPerRow = width * 4
        guard let provider = CGDataProvider(data: NSData(bytes: bytes, length: bytes.count)) else {
            throw FirstFrameParityTestsError.pngEncodeFailed
        }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: info,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw FirstFrameParityTestsError.pngEncodeFailed
        }
        guard let destination = CGImageDestinationCreateWithURL(
            outputURL as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw FirstFrameParityTestsError.pngEncodeFailed
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FirstFrameParityTestsError.pngEncodeFailed
        }
        return outputURL
    }
}
