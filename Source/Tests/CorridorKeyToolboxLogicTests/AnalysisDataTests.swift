//
//  AnalysisDataTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Exercises the NSDictionary round-trip that the FxPlug custom parameter
//  uses for on-disk persistence, plus the frame-index math the render path
//  relies on to locate the right matte for the current play-head.
//

import Foundation
import CoreMedia
import Testing
@testable import CorridorKeyToolboxLogic

@Suite("AnalysisData")
struct AnalysisDataTests {

    @Test("Dictionary round-trip preserves every field")
    func dictionaryRoundTrip() {
        let matteA = Data([0x01, 0x02, 0x03])
        let matteB = Data([0x04, 0x05])
        let original = AnalysisData(
            schemaVersion: AnalysisData.currentSchemaVersion,
            frameDuration: CMTime(value: 1, timescale: 30),
            firstFrameTime: CMTime(value: 0, timescale: 30),
            frameCount: 2,
            analyzedCount: 2,
            screenColorRaw: 0,
            qualityModeRaw: QualityMode.maximum2048.rawValue,
            inferenceResolution: 1024,
            matteWidth: 1024,
            matteHeight: 1024,
            mattes: [0: matteA, 1: matteB]
        )
        let dict = original.asParameterDictionary()
        let roundTripped = AnalysisData.fromParameterDictionary(dict)
        let restored = try? #require(roundTripped)

        #expect(restored?.schemaVersion == AnalysisData.currentSchemaVersion)
        #expect(restored?.frameCount == 2)
        #expect(restored?.analyzedCount == 2)
        #expect(restored?.inferenceResolution == 1024)
        #expect(restored?.matteWidth == 1024)
        #expect(restored?.matteHeight == 1024)
        #expect(restored?.mattes[0] == matteA)
        #expect(restored?.mattes[1] == matteB)
        #expect(restored?.qualityModeRaw == QualityMode.maximum2048.rawValue)
        #expect(CMTimeCompare(restored?.frameDuration ?? .invalid, CMTime(value: 1, timescale: 30)) == 0)
        #expect(CMTimeCompare(restored?.firstFrameTime ?? .invalid, CMTime(value: 0, timescale: 30)) == 0)
    }

    @Test("Schema version mismatch rejects the dictionary")
    func schemaVersionMismatch() {
        let dict = NSMutableDictionary()
        dict[AnalysisDataKey.schemaVersion] = NSNumber(value: AnalysisData.currentSchemaVersion + 99)
        dict[AnalysisDataKey.frameCount] = NSNumber(value: 1)
        #expect(AnalysisData.fromParameterDictionary(dict) == nil)
    }

    @Test("Nil dictionary decodes as nil")
    func nilDictionaryReturnsNil() {
        #expect(AnalysisData.fromParameterDictionary(nil) == nil)
    }

    @Test("Frame index maps render time to the right slot")
    func frameIndexMapping() {
        let data = AnalysisData(
            schemaVersion: AnalysisData.currentSchemaVersion,
            frameDuration: CMTime(value: 1, timescale: 30),
            firstFrameTime: CMTime(value: 0, timescale: 30),
            frameCount: 10,
            analyzedCount: 10,
            screenColorRaw: 0,
            qualityModeRaw: QualityMode.draft512.rawValue,
            inferenceResolution: 512,
            matteWidth: 512,
            matteHeight: 512,
            mattes: [:]
        )
        #expect(data.frameIndex(for: CMTime(value: 0, timescale: 30)) == 0)
        #expect(data.frameIndex(for: CMTime(value: 5, timescale: 30)) == 5)
        #expect(data.frameIndex(for: CMTime(value: 9, timescale: 30)) == 9)
        #expect(data.frameIndex(for: CMTime(value: 10, timescale: 30)) == nil)
        #expect(data.frameIndex(for: CMTime(value: -1, timescale: 30)) == nil)
    }

    @Test("Matte lookup returns the entry for the requested time")
    func matteLookup() {
        let data = AnalysisData(
            schemaVersion: AnalysisData.currentSchemaVersion,
            frameDuration: CMTime(value: 1, timescale: 30),
            firstFrameTime: CMTime(value: 0, timescale: 30),
            frameCount: 3,
            analyzedCount: 2,
            screenColorRaw: 0,
            qualityModeRaw: QualityMode.draft512.rawValue,
            inferenceResolution: 512,
            matteWidth: 512,
            matteHeight: 512,
            mattes: [
                0: Data([0xAA]),
                2: Data([0xCC])
            ]
        )
        #expect(data.matte(at: CMTime(value: 0, timescale: 30)) == Data([0xAA]))
        #expect(data.matte(at: CMTime(value: 1, timescale: 30)) == nil)
        #expect(data.matte(at: CMTime(value: 2, timescale: 30)) == Data([0xCC]))
        #expect(data.matte(at: CMTime(value: 3, timescale: 30)) == nil)
    }
}
