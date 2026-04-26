//
//  PluginStateDataTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Covers the JSON hand-off contract between pluginState and render. FxPlug
//  demands a fresh encoded blob per call, and saved FCP documents survive
//  across plug-in versions, so the decoder must tolerate missing keys that
//  existed in earlier releases (e.g. the removed temporalSmoothing field).
//

import Foundation
import Testing
@testable import CorridorKeyToolboxLogic

@Suite("PluginStateData")
struct PluginStateDataTests {

    @Test("Round-trip preserves every field")
    func roundTripPreservesFields() throws {
        let matte = Data([0xFE, 0xED, 0xFA, 0xCE, 0xCA, 0xFE])
        let original = PluginStateData(
            screenColor: .blue,
            qualityMode: .ultra1536,
            autoSubjectHintEnabled: false,
            sourcePassthroughEnabled: false,
            passthroughErodeNormalized: 12,
            passthroughBlurNormalized: 18,
            alphaBlackPoint: 0.12,
            alphaWhitePoint: 0.88,
            alphaErodeNormalized: -2,
            alphaSoftnessNormalized: 1.5,
            alphaGamma: 1.4,
            autoDespeckleEnabled: true,
            despeckleSize: 850,
            despillStrength: 0.75,
            spillMethod: .doubleLimit,
            temporalStabilityEnabled: false,
            temporalStabilityStrength: 0.77,
            outputMode: .foregroundPlusMatte,
            upscaleMethod: .lanczos,
            renderQualityLevel: 3,
            longEdgeBaseline: 1920,
            destinationLongEdgePixels: 3840,
            cachedMatteBlob: matte,
            cachedMatteInferenceResolution: 1024
        )
        let encoded = try original.encodedForHost()
        let decoded = PluginStateData.decoded(from: encoded)

        #expect(decoded.screenColor == .blue)
        #expect(decoded.qualityMode == .ultra1536)
        #expect(decoded.autoSubjectHintEnabled == false)
        #expect(decoded.sourcePassthroughEnabled == false)
        #expect(decoded.passthroughErodeNormalized == 12)
        #expect(decoded.passthroughBlurNormalized == 18)
        #expect(decoded.alphaBlackPoint == 0.12)
        #expect(decoded.alphaWhitePoint == 0.88)
        #expect(decoded.alphaErodeNormalized == -2)
        #expect(decoded.alphaSoftnessNormalized == 1.5)
        #expect(decoded.alphaGamma == 1.4)
        #expect(decoded.autoDespeckleEnabled == true)
        #expect(decoded.despeckleSize == 850)
        #expect(decoded.despillStrength == 0.75)
        #expect(decoded.spillMethod == .doubleLimit)
        #expect(decoded.temporalStabilityEnabled == false)
        #expect(decoded.temporalStabilityStrength == 0.77)
        #expect(decoded.outputMode == .foregroundPlusMatte)
        #expect(decoded.upscaleMethod == .lanczos)
        #expect(decoded.renderQualityLevel == 3)
        #expect(decoded.longEdgeBaseline == 1920)
        #expect(decoded.destinationLongEdgePixels == 3840)
        #expect(decoded.cachedMatteBlob == matte)
        #expect(decoded.cachedMatteInferenceResolution == 1024)
    }

    @Test("Temporal stability defaults preserve backward compatibility")
    func temporalStabilityDefaults() {
        let defaults = PluginStateData()
        // Defaults OFF in v1.0 — the feature is sound but on subjects
        // with rapid hand or hair motion the gate occasionally lets a
        // partial blend through which can soften legitimate detail
        // for one or two frames. Surface as opt-in so users with
        // visible flicker can enable it per-clip.
        #expect(defaults.temporalStabilityEnabled == false)
        #expect(defaults.temporalStabilityStrength == 0.5)

        // A blob written before this key existed must still decode
        // cleanly and inherit the new default — otherwise projects
        // saved by an earlier build would refuse to open.
        let decoded = PluginStateData.decoded(from: NSData())
        #expect(decoded.temporalStabilityEnabled == false)
        #expect(decoded.temporalStabilityStrength == 0.5)
    }

    @Test("Empty blob falls back to defaults")
    func emptyBlobDecodesDefaults() {
        let decoded = PluginStateData.decoded(from: NSData())
        let defaults = PluginStateData()
        #expect(decoded.qualityMode == defaults.qualityMode)
        #expect(decoded.screenColor == defaults.screenColor)
        #expect(decoded.despillStrength == defaults.despillStrength)
        #expect(decoded.cachedMatteBlob == nil)
        #expect(decoded.cachedMatteInferenceResolution == 0)
    }

    @Test("Garbage blob falls back to defaults without throwing")
    func garbageBlobDecodesDefaults() {
        let garbage = NSData(data: Data([0xDE, 0xAD, 0xBE, 0xEF]))
        let decoded = PluginStateData.decoded(from: garbage)
        let defaults = PluginStateData()
        #expect(decoded.qualityMode == defaults.qualityMode)
        #expect(decoded.despillStrength == defaults.despillStrength)
    }

    @Test("Round-trip without a cached matte preserves nil")
    func roundTripWithoutCachedMatte() throws {
        let original = PluginStateData(screenColor: .green, qualityMode: .high1024)
        let encoded = try original.encodedForHost()
        let decoded = PluginStateData.decoded(from: encoded)
        #expect(decoded.cachedMatteBlob == nil)
        #expect(decoded.cachedMatteInferenceResolution == 0)
    }

    @Test("Auto subject hint defaults to disabled and survives round-trip")
    func autoSubjectHintDefaultsAndRoundTrip() throws {
        let defaults = PluginStateData()
        // Defaults OFF in v1.0: the MLX bridge was trained on the
        // soft gradient green-bias hint, and Vision returns a binary
        // mask which is structurally different. Opt-in only.
        #expect(defaults.autoSubjectHintEnabled == false)

        // A blob written before this key existed must still decode
        // cleanly and inherit the safe default.
        let legacyDecoded = PluginStateData.decoded(from: NSData())
        #expect(legacyDecoded.autoSubjectHintEnabled == false)

        // On explicitly should round-trip.
        let original = PluginStateData(autoSubjectHintEnabled: true)
        let encoded = try original.encodedForHost()
        let decoded = PluginStateData.decoded(from: encoded)
        #expect(decoded.autoSubjectHintEnabled == true)
    }

    @Test("destinationPixelRadius scales by clip size")
    func destinationPixelRadiusScales() {
        var state = PluginStateData()
        state.longEdgeBaseline = 1920
        state.destinationLongEdgePixels = 3840
        // 1920 baseline → 10-unit control yields 20px on a 4K clip.
        #expect(abs(state.destinationPixelRadius(fromNormalized: 10) - 20) < 0.001)

        state.destinationLongEdgePixels = 1920
        #expect(abs(state.destinationPixelRadius(fromNormalized: 10) - 10) < 0.001)
    }
}
