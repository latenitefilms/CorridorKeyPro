//
//  ScreenColorAutoDetector.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Picks the best Screen Colour for a freshly-imported clip by scoring
//  the first frame for blue-dominance vs green-dominance. The user can
//  still override manually after import — this is a "good first guess"
//  so a blue-screen project doesn't have to flip the popup before the
//  preview shows a usable matte.
//
//  The detector is intentionally cheap. We sample a stride-decimated
//  ~64×64 grid out of the first frame's pixel buffer, accumulate the
//  per-pixel "screen-bias" score for both colours, and return whichever
//  total is larger. A small confidence margin prevents the result from
//  flapping on shots where neither colour dominates (text, neutral
//  backgrounds) — the editor falls back to the user's last-used colour
//  in that case.
//

import Foundation
import CoreVideo
import CoreGraphics

enum ScreenColorAutoDetector {

    /// Returns the recommended screen colour for the supplied frame.
    /// `fallback` is used when neither green nor blue dominates strongly
    /// enough to be confident — typically the user's last-saved choice
    /// from `EditorPreferences` so reopening a non-screen reference clip
    /// doesn't flip the inspector unnecessarily.
    static func detect(
        firstFrame pixelBuffer: CVPixelBuffer,
        fallback: ScreenColor
    ) -> ScreenColor {
        // The detector runs on whatever pixel format the decoder gave
        // us; refuse to over-engineer for every conceivable format and
        // bail out (returning the fallback) for anything we don't have
        // a fast-path reader for. The standard hand-off in
        // `VideoSource.makeFrame` returns either RGBAHalf or BGRA8.
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        switch format {
        case kCVPixelFormatType_32BGRA:
            return scoreBGRA8(pixelBuffer: pixelBuffer, fallback: fallback)
        case kCVPixelFormatType_64RGBAHalf:
            return scoreRGBAHalf(pixelBuffer: pixelBuffer, fallback: fallback)
        default:
            return fallback
        }
    }

    /// Number of samples per axis. Sixty-four points × sixty-four points
    /// is enough to dominate sampling noise from a single foreground
    /// pixel without paying for a full-resolution scan.
    private static let sampleAxis = 64

    /// Bias floor — values below this threshold are too neutral to
    /// count toward either colour's score. Stops a slightly-tinted
    /// chair from voting for "green" and the like.
    private static let biasFloor: Float = 0.08

    /// Confidence margin between the winner and runner-up before we
    /// commit. Keeps the auto-detect from snapping to the wrong colour
    /// on shots that have a little of both (e.g. mixed lighting on a
    /// half-and-half cyc wall).
    private static let confidenceMargin: Float = 1.25

    private static func scoreBGRA8(
        pixelBuffer: CVPixelBuffer,
        fallback: ScreenColor
    ) -> ScreenColor {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return fallback
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)

        var greenScore: Float = 0
        var blueScore: Float = 0
        sampleGrid(width: width, height: height) { sampleX, sampleY in
            let offset = sampleY * bytesPerRow + sampleX * 4
            // BGRA layout — the Metal-friendly format the bridge
            // produces.
            let blue = Float(pointer[offset + 0]) / 255
            let green = Float(pointer[offset + 1]) / 255
            let red = Float(pointer[offset + 2]) / 255
            accumulate(red: red, green: green, blue: blue,
                       greenScore: &greenScore, blueScore: &blueScore)
        }
        return decide(greenScore: greenScore, blueScore: blueScore, fallback: fallback)
    }

    private static func scoreRGBAHalf(
        pixelBuffer: CVPixelBuffer,
        fallback: ScreenColor
    ) -> ScreenColor {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return fallback
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pointer = baseAddress.assumingMemoryBound(to: UInt16.self)
        let bytesPerPixel = 8
        let pixelsPerRow = bytesPerRow / bytesPerPixel

        var greenScore: Float = 0
        var blueScore: Float = 0
        sampleGrid(width: width, height: height) { sampleX, sampleY in
            // Half-float row stride: bytesPerRow ÷ 8 bytes per pixel.
            let pixelOffset = sampleY * pixelsPerRow + sampleX
            let base = pixelOffset * 4
            let red = Self.halfToFloat(pointer[base + 0])
            let green = Self.halfToFloat(pointer[base + 1])
            let blue = Self.halfToFloat(pointer[base + 2])
            accumulate(red: red, green: green, blue: blue,
                       greenScore: &greenScore, blueScore: &blueScore)
        }
        return decide(greenScore: greenScore, blueScore: blueScore, fallback: fallback)
    }

    /// Visits a stride-sampled grid over the full frame. Each callback
    /// receives `(x, y)` integer coordinates clamped inside the frame
    /// dimensions. Skips the call entirely on degenerate frames.
    private static func sampleGrid(
        width: Int,
        height: Int,
        body: (_ x: Int, _ y: Int) -> Void
    ) {
        guard width > 0, height > 0 else { return }
        let denom = max(sampleAxis, 1)
        for sampleY in 0..<sampleAxis {
            // Centre each sample inside its grid cell so the corner
            // pixel doesn't dominate on small frames (e.g. a 256-pixel
            // proxy thumbnail).
            let pixelY = min(height - 1, ((sampleY * 2 + 1) * height) / (denom * 2))
            for sampleX in 0..<sampleAxis {
                let pixelX = min(width - 1, ((sampleX * 2 + 1) * width) / (denom * 2))
                body(pixelX, pixelY)
            }
        }
    }

    /// Adds the pixel's per-colour bias to the running totals. The
    /// `biasFloor` keeps near-neutral pixels (foreground subjects, set
    /// dressing) from contributing noise.
    private static func accumulate(
        red: Float,
        green: Float,
        blue: Float,
        greenScore: inout Float,
        blueScore: inout Float
    ) {
        let greenBias = green - max(red, blue)
        let blueBias = blue - max(red, green)
        if greenBias > biasFloor { greenScore += greenBias }
        if blueBias > biasFloor { blueScore += blueBias }
    }

    /// Picks the stronger colour, requiring a `confidenceMargin`
    /// advantage so genuinely ambiguous frames fall back to the user's
    /// last choice.
    private static func decide(
        greenScore: Float,
        blueScore: Float,
        fallback: ScreenColor
    ) -> ScreenColor {
        if greenScore > blueScore * confidenceMargin {
            return .green
        }
        if blueScore > greenScore * confidenceMargin {
            return .blue
        }
        return fallback
    }

    /// IEEE-754 half-precision → single-precision conversion. Apple's
    /// Accelerate framework has `vImageConvert_Planar16FtoPlanarF`
    /// for this, but that needs scratch buffers and a planar copy —
    /// overkill for a 4 096-sample detection pass. The bit-shuffling
    /// here is the standard 5-bit exponent + 10-bit mantissa expansion
    /// to 8-bit + 23-bit. Subnormals and infinities are handled because
    /// the IEEE rules generalise; NaN flows through as NaN.
    private static func halfToFloat(_ raw: UInt16) -> Float {
        let sign = UInt32(raw & 0x8000) << 16
        let exponent = (raw >> 10) & 0x1F
        let mantissa = UInt32(raw & 0x03FF)

        if exponent == 0 {
            // Zero / subnormal — represent as zero. Subnormals are too
            // small to matter for the screen-colour decision.
            return Float(bitPattern: sign)
        }
        if exponent == 0x1F {
            // Inf / NaN — propagate as Inf so it dominates the running
            // sum and the frame is flagged as undecidable in practice.
            let bits = sign | 0x7F800000 | (mantissa << 13)
            return Float(bitPattern: bits)
        }
        let adjustedExponent = UInt32(exponent) + (127 - 15)
        let bits = sign | (adjustedExponent << 23) | (mantissa << 13)
        return Float(bitPattern: bits)
    }
}
