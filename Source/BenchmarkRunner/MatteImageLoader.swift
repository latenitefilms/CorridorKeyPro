//
//  MatteImageLoader.swift
//  BenchmarkRunner
//
//  Loads a single matte frame from disk into a float-32 grayscale buffer.
//  Supports any format ImageIO understands natively — PNG (including
//  16-bit), TIFF, OpenEXR with supported compression, etc. The benchmark
//  matte sequences shipped as EXR DWAB which macOS can't decode, so the
//  expected flow is to pre-convert them once with `ffmpeg` (see
//  `Benchmarks/README.md`) and point the runner at the resulting PNG or
//  TIFF folder.
//
//  The returned buffer is one Float32 per pixel arranged in row-major
//  order, top-to-bottom, with values clamped to the `[0, 1]` matte range.
//

import Foundation
import CoreImage
import CoreGraphics

enum MatteImageLoader {

    /// Flat float-32 pixel buffer, top-to-bottom, row-major, `width * height`
    /// elements. Decoded-then-released on every call so a streaming temporal
    /// pass doesn't accumulate memory frame-by-frame.
    struct Buffer {
        let pixels: [Float]
        let width: Int
        let height: Int

        var pixelCount: Int { width * height }
    }

    /// Shared CIContext — constructing these is expensive, and `render` is
    /// thread-safe so a single instance amortises across every frame.
    private static let sharedContext: CIContext = {
        CIContext(options: [.useSoftwareRenderer: false])
    }()

    /// Linear sRGB space. Rendering into this colour space skips the gamma
    /// encode that CoreImage would otherwise apply on an sRGB-tagged PNG,
    /// so the raw matte values survive the round-trip unchanged.
    private static let linearRGBColorSpace: CGColorSpace = {
        CGColorSpace(name: CGColorSpace.linearSRGB) ?? CGColorSpaceCreateDeviceRGB()
    }()

    /// Renders the image at `url` into an RGBAf buffer, then extracts the
    /// red channel. Mattes in the benchmark fixtures are single-channel
    /// with R=G=B=matte value, so the red channel alone is authoritative.
    /// Alpha and the other channels are discarded.
    static func load(_ url: URL) throws -> Buffer {
        guard let ciImage = CIImage(contentsOf: url) else {
            throw BenchmarkError.imageLoadFailed(url, underlying: LoaderDetail(stage: "ciImageCreate"))
        }

        let extent = ciImage.extent
        let width = Int(extent.width)
        let height = Int(extent.height)
        guard width > 0, height > 0 else {
            throw BenchmarkError.imageLoadFailed(url, underlying: LoaderDetail(stage: "extentEmpty"))
        }

        let pixelCount = width * height
        // Render into RGBAf in a linear colour space so matte values survive
        // unchanged. We then walk the buffer once, pulling out the red
        // channel into a flat float array for downstream analysis.
        var rgbaBuffer = [SIMD4<Float>](
            repeating: SIMD4<Float>(0, 0, 0, 0),
            count: pixelCount
        )
        rgbaBuffer.withUnsafeMutableBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress else { return }
            sharedContext.render(
                ciImage,
                toBitmap: baseAddress,
                rowBytes: width * MemoryLayout<SIMD4<Float>>.stride,
                bounds: extent,
                format: .RGBAf,
                colorSpace: linearRGBColorSpace
            )
        }

        // Core Image renders with a bottom-up origin; flip to top-down and
        // extract the red channel as the matte value in one pass so
        // downstream temporal math treats pixel (0,0) as the top-left.
        var flipped = [Float](repeating: 0, count: pixelCount)
        for row in 0..<height {
            let sourceRow = height - 1 - row
            let sourceStart = sourceRow * width
            let destinationStart = row * width
            for column in 0..<width {
                flipped[destinationStart + column] = rgbaBuffer[sourceStart + column].x.clampedToMatteRange
            }
        }
        return Buffer(pixels: flipped, width: width, height: height)
    }
}

/// Diagnostic payload used when the loader fails — surfaces the exact stage
/// of the pipeline that produced the error so test fixtures can assert on it.
struct LoaderDetail: Error, CustomStringConvertible {
    let stage: String
    var description: String { "at stage \(stage)" }
}

private extension Float {
    /// Clamp helper. We intentionally treat NaN as zero (matte outside the
    /// valid range) instead of propagating — the benchmark needs a number.
    var clampedToMatteRange: Float {
        if isNaN { return 0 }
        if self < 0 { return 0 }
        if self > 1 { return 1 }
        return self
    }
}
