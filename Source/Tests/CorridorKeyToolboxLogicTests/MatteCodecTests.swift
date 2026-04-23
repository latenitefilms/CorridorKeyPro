//
//  MatteCodecTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Codec invariants for the persisted matte blob. Anything a user saves in
//  their Final Cut Pro Library flows through this code, so small round-trip
//  errors would accumulate into visible keying drift on playback.
//

import Foundation
import Testing
@testable import CorridorKeyToolboxLogic

@Suite("MatteCodec")
struct MatteCodecTests {

    @Test("Round-trip of a smooth alpha preserves values within half-float epsilon")
    func roundTripSmoothAlpha() throws {
        let width = 64
        let height = 64
        var alpha = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                alpha[y * width + x] = Float(x) / Float(max(width - 1, 1))
            }
        }

        let encoded = try MatteCodec.encode(alpha: alpha, width: width, height: height)
        let decoded = try #require(MatteCodec.decode(encoded))
        #expect(decoded.width == width)
        #expect(decoded.height == height)
        #expect(decoded.alpha.count == alpha.count)
        for index in alpha.indices {
            #expect(abs(decoded.alpha[index] - alpha[index]) < 1e-3)
        }
    }

    @Test("Values are clamped to 0...1 on encode")
    func clampingOnEncode() throws {
        let alpha: [Float] = [-0.5, 0, 0.25, 1.0, 1.5]
        let encoded = try MatteCodec.encode(alpha: alpha, width: 5, height: 1)
        let decoded = try #require(MatteCodec.decode(encoded))
        #expect(decoded.alpha[0] == 0)
        #expect(decoded.alpha[1] == 0)
        #expect(abs(decoded.alpha[2] - 0.25) < 1e-3)
        #expect(abs(decoded.alpha[3] - 1.0) < 1e-3)
        #expect(decoded.alpha[4] == 1.0)
    }

    @Test("Compression wins versus raw half-float bytes")
    func compressionIsEffective() throws {
        let width = 256
        let height = 256
        // Uniform matte — zlib should collapse it dramatically.
        let alpha = [Float](repeating: 0.5, count: width * height)
        let encoded = try MatteCodec.encode(alpha: alpha, width: width, height: height)
        let rawHalfBytes = width * height * MemoryLayout<UInt16>.size
        #expect(encoded.count < rawHalfBytes / 4)
    }

    @Test("Mismatched size throws on encode")
    func encodeMismatchThrows() {
        let alpha: [Float] = [0, 1, 0]
        #expect(throws: MatteCodecError.self) {
            try MatteCodec.encode(alpha: alpha, width: 4, height: 1)
        }
    }

    @Test("Truncated payload decodes as nil")
    func truncatedPayloadReturnsNil() throws {
        let alpha: [Float] = [0, 0.5, 1, 0.25]
        let encoded = try MatteCodec.encode(alpha: alpha, width: 4, height: 1)
        let truncated = encoded.prefix(6) // drop into the compressed payload
        #expect(MatteCodec.decode(Data(truncated)) == nil)
    }

    @Test("Missing magic header decodes as nil")
    func missingMagicReturnsNil() {
        var blob = Data()
        blob.append(contentsOf: [0x00, 0x00, 0x00, 0x00])  // wrong magic
        var width = Int32(1).littleEndian
        var height = Int32(1).littleEndian
        withUnsafeBytes(of: &width) { blob.append(contentsOf: $0) }
        withUnsafeBytes(of: &height) { blob.append(contentsOf: $0) }
        blob.append(0x78)
        blob.append(0x9C)
        #expect(MatteCodec.decode(blob) == nil)
    }

    @Test("Float/half conversion covers boundary inputs",
          arguments: [0.0 as Float, 0.25, 0.5, 0.9999, 1.0])
    func halfFloatBoundaries(value: Float) {
        let half = MatteCodec.floatToHalf(value)
        let restored = MatteCodec.halfToFloat(half)
        #expect(abs(restored - value) < 1e-3)
    }
}
