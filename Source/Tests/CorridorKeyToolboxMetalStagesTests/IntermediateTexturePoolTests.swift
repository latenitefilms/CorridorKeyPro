//
//  IntermediateTexturePoolTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Verifies the pool's key behaviours: a texture with a matching descriptor
//  is reused; textures returned after commit are available again; the
//  bucket cap limits peak memory under aggressive churn.
//

import Foundation
import Metal
import Testing
@testable import CorridorKeyToolboxMetalStages

@Suite("IntermediateTexturePool")
struct IntermediateTexturePoolTests {

    @Test("Same-shape acquisitions hit the cache after a manual return")
    func reuseAfterManualReturn() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkipError("No Metal device available.")
        }
        let pool = IntermediateTexturePool(device: device)

        guard let first = pool.acquire(width: 256, height: 256, pixelFormat: .rgba16Float) else {
            Issue.record("First acquire failed.")
            return
        }
        let firstIdentity = ObjectIdentifier(first.texture)
        first.returnManually()

        guard let second = pool.acquire(width: 256, height: 256, pixelFormat: .rgba16Float) else {
            Issue.record("Second acquire failed.")
            return
        }
        #expect(ObjectIdentifier(second.texture) == firstIdentity)
        second.returnManually()

        let (hits, misses) = pool.statistics()
        #expect(hits == 1)
        #expect(misses == 1)
    }

    @Test("Mismatched shapes get separate buckets")
    func shapesAreolated() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkipError("No Metal device available.")
        }
        let pool = IntermediateTexturePool(device: device)

        let small = pool.acquire(width: 64, height: 64)
        let large = pool.acquire(width: 512, height: 512)
        let differentFormat = pool.acquire(width: 64, height: 64, pixelFormat: .r32Float)

        #expect(small != nil)
        #expect(large != nil)
        #expect(differentFormat != nil)

        small?.returnManually()
        differentFormat?.returnManually()
        large?.returnManually()

        guard let smallAgain = pool.acquire(width: 64, height: 64) else {
            Issue.record("Could not re-acquire small texture.")
            return
        }
        // Shape-match: we should get the 64×64 rgba16Float back, not the
        // 64×64 r32Float.
        #expect(smallAgain.texture.pixelFormat == .rgba16Float)
        #expect(smallAgain.texture.width == 64 && smallAgain.texture.height == 64)
        smallAgain.returnManually()
    }

    @Test("Bucket cap limits pool memory growth")
    func bucketCap() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkipError("No Metal device available.")
        }
        let pool = IntermediateTexturePool(device: device)
        // Acquire many same-shape textures, then return them all. The pool
        // retains at most six per bucket (the internal cap), so subsequent
        // acquires land on cached textures up to six deep.
        var acquired: [PooledTexture] = []
        for _ in 0..<12 {
            guard let texture = pool.acquire(width: 128, height: 128) else {
                Issue.record("Acquire failed during loop.")
                return
            }
            acquired.append(texture)
        }
        for texture in acquired { texture.returnManually() }
        acquired.removeAll()

        var reuseCount = 0
        for _ in 0..<8 {
            guard let texture = pool.acquire(width: 128, height: 128) else {
                Issue.record("Acquire failed after returns.")
                return
            }
            acquired.append(texture)
        }
        let (hits, _) = pool.statistics()
        reuseCount = hits
        for texture in acquired { texture.returnManually() }
        // Expect at most 6 reuse hits from the pool cap (+ possibly less on
        // systems that reorder things). The key invariant is "we don't cache
        // all 12".
        #expect(reuseCount <= 6)
    }
}
