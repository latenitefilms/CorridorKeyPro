//
//  MetalDeviceCacheEntryTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Exercises the weights cache on `MetalDeviceCacheEntry`. Verifies that
//  identical radius + sigma return the same buffer object, and that
//  analytically-correct Gaussian weights come out of the helper.
//

import Foundation
import Metal
import Testing
@testable import CorridorKeyToolboxMetalStages

@Suite("MetalDeviceCacheEntry")
struct MetalDeviceCacheEntryTests {

    @Test("Identical Gaussian weights share a buffer")
    func gaussianWeightsShareBuffer() async throws {
        let entry: MetalDeviceCacheEntry
        do {
            entry = try TestHarness.makeEntry()
        } catch let error as XCTSkipError {
            throw error
        } catch let error as MetalUnavailable {
            throw XCTSkipError(error.description)
        }

        let first = entry.gaussianWeightsBuffer(radius: 5, sigma: 1.8)
        let second = entry.gaussianWeightsBuffer(radius: 5, sigma: 1.8)
        #expect(first != nil)
        #expect(second != nil)
        if let first, let second {
            #expect(ObjectIdentifier(first.buffer) == ObjectIdentifier(second.buffer))
            #expect(first.count == 6) // radius + 1 one-sided taps
        }
    }

    @Test("Nearby sigmas quantise onto the same bucket")
    func nearbySigmasShareBucket() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }
        let a = entry.gaussianWeightsBuffer(radius: 8, sigma: 2.32)
        let b = entry.gaussianWeightsBuffer(radius: 8, sigma: 2.34)
        #expect(a != nil && b != nil)
        if let a, let b {
            #expect(ObjectIdentifier(a.buffer) == ObjectIdentifier(b.buffer))
        }
    }

    @Test("Gaussian weights sum to 1 with matched radius+sigma")
    func gaussianWeightsNormalise() {
        let weights = MetalDeviceCacheEntry.makeGaussianWeights(radius: 6, sigma: 1.5)
        #expect(weights.count == 7)
        // Sum = w[0] + 2*sum(w[1..radius]) (the shader mirrors taps).
        var total: Float = weights[0]
        for index in 1..<weights.count {
            total += weights[index] * 2
        }
        #expect(abs(total - 1.0) < 1e-4)
    }
}
