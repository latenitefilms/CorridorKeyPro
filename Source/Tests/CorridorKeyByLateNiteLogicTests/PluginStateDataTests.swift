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
            hintMode: .automatic,
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
        #expect(decoded.hintMode == .automatic)
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
        // Default ON in v1.0 — alongside Light Wrap, Edge Decontaminate,
        // Auto Despeckle, and Apple Vision hinting, this lands users on
        // a comp-ready key on first frame. Users tuning a tricky shot
        // can flip it off via the inspector.
        #expect(defaults.temporalStabilityEnabled == true)
        #expect(defaults.temporalStabilityStrength == 0.5)

        // Empty blobs hit the default branch which now returns an
        // enabled stabiliser, matching the new product default.
        let decoded = PluginStateData.decoded(from: NSData())
        #expect(decoded.temporalStabilityEnabled == true)
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

    @Test("Hint mode defaults to Apple Vision and survives round-trip")
    func hintModeDefaultsAndRoundTrip() throws {
        let defaults = PluginStateData()
        // Default in v1.0: Vision's foreground prior beats the
        // chroma prior on most footage, and the MLX network handles
        // the harder boundaries Vision produces well enough that
        // shipping it as the default is the right call.
        #expect(defaults.hintMode == .appleVision)

        // Empty blob falls back to the default mode.
        let legacyDecoded = PluginStateData.decoded(from: NSData())
        #expect(legacyDecoded.hintMode == .appleVision)

        // Manual should round-trip.
        let manual = PluginStateData(screenColor: .green, qualityMode: .high1024, hintMode: .manual)
        let encoded = try manual.encodedForHost()
        let decoded = PluginStateData.decoded(from: encoded)
        #expect(decoded.hintMode == .manual)

        // Automatic should round-trip.
        let automatic = PluginStateData(screenColor: .green, qualityMode: .high1024, hintMode: .automatic)
        let encodedAutomatic = try automatic.encodedForHost()
        let decodedAutomatic = PluginStateData.decoded(from: encodedAutomatic)
        #expect(decodedAutomatic.hintMode == .automatic)
    }

    @Test("Decoder fallbacks match designated-init defaults")
    func decoderFallbacksMatchInitDefaults() throws {
        // Encode a `PluginStateData` plist that's missing every key
        // FxPlug might omit when reading back a project saved with an
        // older build, then verify the decoded value matches the
        // current product default. Drift between the designated init
        // and `init(from decoder:)` is what made the FxPlug ship with
        // a fundamentally different look from the Standalone Editor
        // for users opening pre-v1.0 libraries — the decoder gave
        // them stale "off" defaults for the post-process toggles
        // while the editor (constructing `PluginStateData()`) ran
        // with the comp-ready "on" defaults.
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        struct LegacyState: Encodable {
            // Only the four genuinely required fields — every other
            // key is intentionally absent so `decodeIfPresent` falls
            // through to its default.
            var screenColor: ScreenColor
            var qualityMode: QualityMode
            var hintMode: HintMode
        }
        let legacy = LegacyState(
            screenColor: .green,
            qualityMode: .automatic,
            hintMode: .appleVision
        )
        let blob = try encoder.encode(legacy)
        let decoded = PluginStateData.decoded(from: NSData(data: blob))
        let defaults = PluginStateData()

        #expect(decoded.sourcePassthroughEnabled == defaults.sourcePassthroughEnabled)
        #expect(decoded.autoDespeckleEnabled == defaults.autoDespeckleEnabled)
        #expect(decoded.lightWrapEnabled == defaults.lightWrapEnabled)
        #expect(decoded.edgeDecontaminateEnabled == defaults.edgeDecontaminateEnabled)
        #expect(decoded.temporalStabilityEnabled == defaults.temporalStabilityEnabled)
        #expect(decoded.spillMethod == defaults.spillMethod)
        #expect(decoded.refinerStrength == defaults.refinerStrength)
        #expect(decoded.lightWrapStrength == defaults.lightWrapStrength)
        #expect(decoded.lightWrapRadius == defaults.lightWrapRadius)
        #expect(decoded.edgeDecontaminateStrength == defaults.edgeDecontaminateStrength)
        #expect(decoded.temporalStabilityStrength == defaults.temporalStabilityStrength)
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
