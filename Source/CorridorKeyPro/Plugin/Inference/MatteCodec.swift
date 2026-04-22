//
//  MatteCodec.swift
//  Corridor Key Toolbox
//
//  Encodes and decodes single-channel alpha mattes for embedding inside the
//  plug-in's FxPlug custom parameter. The matte is the only thing we persist
//  per frame — the neural network's foreground is reconstructed at render
//  time from the original source, so saving it would quadruple storage for
//  no visible quality win once despill runs.
//
//  Layout of a compressed frame:
//
//      magic      "CKM1" (4 bytes, identifies the format)
//      width      Int32  (matte width in pixels)
//      height     Int32  (matte height in pixels)
//      payload    zlib-compressed Float16 buffer (width * height samples)
//
//  Float16 halves memory versus Float32 while staying visually lossless for
//  smooth alpha gradients. Zlib then squeezes typical mattes by another 4–6x.
//

import Foundation
import Compression

enum MatteCodecError: Error, CustomStringConvertible {
    case invalidSize
    case payloadTooShort
    case badMagic
    case compressionFailed
    case decompressionFailed
    case dimensionsOutOfBounds

    var description: String {
        switch self {
        case .invalidSize:
            return "Matte buffer size does not match the requested dimensions."
        case .payloadTooShort:
            return "Compressed matte blob is too short to contain a valid header."
        case .badMagic:
            return "Compressed matte blob does not carry the expected CorridorKey header."
        case .compressionFailed:
            return "Could not compress the alpha matte."
        case .decompressionFailed:
            return "Could not decompress the alpha matte."
        case .dimensionsOutOfBounds:
            return "Matte dimensions are outside the range the plug-in allows."
        }
    }
}

/// Codec for alpha-only matte persistence inside the FxPlug custom parameter.
/// Values are clamped to `[0, 1]` before encoding — the plug-in never stores
/// negative or super-bright mattes so narrowing here wins space without any
/// perceptible quality loss.
enum MatteCodec {

    /// 4-byte magic prefix. Changing this string bumps the on-disk format
    /// version; older blobs will fail to decode and `decoded(_:)` returns nil
    /// so the render path can fall back to live MLX.
    private static let magic: [UInt8] = Array("CKM1".utf8)

    /// Upper bound on the dimension of a single-channel matte. A 4096² matte
    /// is already 32 MB uncompressed float32, well beyond anything we would
    /// want to persist per frame — clamping here guards against corrupt blobs
    /// that try to allocate huge buffers during decode.
    private static let maximumSide: Int = 4096

    /// Encodes a float alpha buffer into a compact byte blob. `alpha.count`
    /// must equal `width * height`; values are clamped to 0..1.
    static func encode(alpha: [Float], width: Int, height: Int) throws -> Data {
        guard width > 0, height > 0,
              width <= maximumSide, height <= maximumSide
        else {
            throw MatteCodecError.dimensionsOutOfBounds
        }
        guard alpha.count == width * height else {
            throw MatteCodecError.invalidSize
        }

        var halfBuffer = [UInt16](repeating: 0, count: alpha.count)
        for index in alpha.indices {
            let clamped = min(max(alpha[index], 0), 1)
            halfBuffer[index] = Self.floatToHalf(clamped)
        }

        let rawBytes = halfBuffer.withUnsafeBufferPointer { pointer in
            Data(buffer: pointer)
        }
        guard let compressed = compress(rawBytes) else {
            throw MatteCodecError.compressionFailed
        }

        var output = Data()
        output.reserveCapacity(magic.count + 8 + compressed.count)
        output.append(contentsOf: magic)
        var widthLE = Int32(width).littleEndian
        var heightLE = Int32(height).littleEndian
        withUnsafeBytes(of: &widthLE) { output.append(contentsOf: $0) }
        withUnsafeBytes(of: &heightLE) { output.append(contentsOf: $0) }
        output.append(compressed)
        return output
    }

    /// Decodes a previously-encoded blob into a float alpha buffer plus its
    /// dimensions. Returns nil for unknown magic or decompression failure —
    /// callers treat that as "no cache available" and fall back to live MLX.
    static func decode(_ blob: Data) -> (alpha: [Float], width: Int, height: Int)? {
        guard blob.count >= magic.count + 8 else { return nil }
        let magicSlice = blob.prefix(magic.count)
        guard Array(magicSlice) == magic else { return nil }

        let headerOffset = magic.count
        let width = Int(blob.withUnsafeBytes { pointer -> Int32 in
            pointer.load(fromByteOffset: headerOffset, as: Int32.self).littleEndian
        })
        let height = Int(blob.withUnsafeBytes { pointer -> Int32 in
            pointer.load(fromByteOffset: headerOffset + 4, as: Int32.self).littleEndian
        })
        guard width > 0, height > 0, width <= maximumSide, height <= maximumSide else {
            return nil
        }

        let payload = blob.suffix(from: headerOffset + 8)
        let expectedRawBytes = width * height * MemoryLayout<UInt16>.size
        guard let decompressed = decompress(Data(payload), expectedBytes: expectedRawBytes),
              decompressed.count == expectedRawBytes
        else {
            return nil
        }

        var alpha = [Float](repeating: 0, count: width * height)
        decompressed.withUnsafeBytes { rawPointer in
            guard let base = rawPointer.baseAddress?.assumingMemoryBound(to: UInt16.self) else { return }
            for index in 0..<(width * height) {
                alpha[index] = Self.halfToFloat(base[index])
            }
        }
        return (alpha, width, height)
    }

    // MARK: - Compression helpers

    private static func compress(_ source: Data) -> Data? {
        return source.withUnsafeBytes { rawPointer -> Data? in
            guard let sourcePointer = rawPointer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return nil
            }
            let destinationCapacity = max(source.count + 128, 256)
            let destination = UnsafeMutablePointer<UInt8>.allocate(capacity: destinationCapacity)
            defer { destination.deallocate() }
            let compressedSize = compression_encode_buffer(
                destination, destinationCapacity,
                sourcePointer, source.count,
                nil, COMPRESSION_ZLIB
            )
            guard compressedSize > 0 else { return nil }
            return Data(bytes: destination, count: compressedSize)
        }
    }

    private static func decompress(_ source: Data, expectedBytes: Int) -> Data? {
        guard expectedBytes > 0 else { return nil }
        return source.withUnsafeBytes { rawPointer -> Data? in
            guard let sourcePointer = rawPointer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return nil
            }
            let destination = UnsafeMutablePointer<UInt8>.allocate(capacity: expectedBytes)
            defer { destination.deallocate() }
            let size = compression_decode_buffer(
                destination, expectedBytes,
                sourcePointer, source.count,
                nil, COMPRESSION_ZLIB
            )
            guard size == expectedBytes else { return nil }
            return Data(bytes: destination, count: size)
        }
    }

    // MARK: - IEEE 754 half conversion

    /// Converts a finite Float32 to IEEE 754 binary16 (round-to-nearest-even).
    /// Out-of-range inputs are clamped to 0 / max finite half / infinity as
    /// appropriate. We don't emit NaNs — the callers already clamp to [0, 1].
    static func floatToHalf(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int32((bits >> 23) & 0xFF) - 127 + 15
        let mantissa = bits & 0x7FFFFF

        if exponent <= 0 {
            if exponent < -10 {
                return sign
            }
            let combinedMantissa = mantissa | 0x800000
            let shift = 14 - exponent
            var half = combinedMantissa >> shift
            if (combinedMantissa >> (shift - 1)) & 1 != 0 {
                half += 1
            }
            return sign | UInt16(half & 0xFFFF)
        }
        if exponent >= 31 {
            return sign | 0x7C00
        }
        var half = UInt32(exponent) << 10 | (mantissa >> 13)
        if mantissa & 0x1000 != 0 {
            half += 1
        }
        return sign | UInt16(half & 0xFFFF)
    }

    /// Converts IEEE 754 binary16 back to Float32. Subnormals and infinities
    /// are handled so round-tripped values land exactly on the half-precision
    /// grid.
    static func halfToFloat(_ half: UInt16) -> Float {
        let sign = UInt32(half & 0x8000) << 16
        var exponent = Int32((half >> 10) & 0x1F)
        var mantissa = UInt32(half & 0x3FF)

        if exponent == 0 {
            if mantissa == 0 {
                return Float(bitPattern: sign)
            }
            while (mantissa & 0x400) == 0 {
                mantissa <<= 1
                exponent -= 1
            }
            exponent += 1
            mantissa &= ~UInt32(0x400)
        } else if exponent == 31 {
            return Float(bitPattern: sign | 0x7F800000 | (mantissa << 13))
        }

        let resultExponent = UInt32(exponent + (127 - 15)) << 23
        return Float(bitPattern: sign | resultExponent | (mantissa << 13))
    }
}
