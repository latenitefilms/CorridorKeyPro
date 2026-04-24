//
//  PluginStateData.swift
//  Corridor Key Toolbox
//
//  Immutable snapshot of every parameter value at a specific render time.
//  Created in `pluginState(_:at:quality:)` and handed back to the plugin in
//  every subsequent call. FxPlug requires a fresh instance per call because
//  the host may read state from multiple threads at once.
//

import Foundation
import CoreMedia

/// A fully resolved render state. Contains everything the renderer and analyser
/// need so that no parameter API calls happen off of the main FxPlug thread.
struct PluginStateData: Codable, Sendable {
    // Key setup
    var screenColor: ScreenColor
    var qualityMode: QualityMode

    // Interior detail
    var sourcePassthroughEnabled: Bool
    var passthroughErodeNormalized: Double
    var passthroughBlurNormalized: Double

    // Matte
    var alphaBlackPoint: Double
    var alphaWhitePoint: Double
    var alphaErodeNormalized: Double
    var alphaSoftnessNormalized: Double
    var alphaGamma: Double
    var autoDespeckleEnabled: Bool
    var despeckleSize: Int
    var refinerStrength: Double

    // Edge and spill
    var despillStrength: Double
    var spillMethod: SpillMethod

    // Edge refinement (Phase 4 — light wrap + edge decontamination)
    var lightWrapEnabled: Bool
    var lightWrapStrength: Double
    var lightWrapRadius: Double
    var edgeDecontaminateEnabled: Bool
    var edgeDecontaminateStrength: Double

    // Temporal stability (Phase 1 — analysis-pass matte EMA with motion gate)
    var temporalStabilityEnabled: Bool
    var temporalStabilityStrength: Double

    // Output
    var outputMode: OutputMode

    // Performance and quality
    var upscaleMethod: UpscaleMethod

    // Runtime
    var renderQualityLevel: Int

    /// Long edge used by the parameter panel to scale pixel-space controls to
    /// the current clip. Mirrors the OFX plugin's 1920px baseline behaviour.
    var longEdgeBaseline: Double

    /// Actual long edge of the destination image at render time, used to
    /// translate normalised radii into destination pixels.
    var destinationLongEdgePixels: Int

    /// When the FxAnalyzer pass has cached this frame's matte, we embed the
    /// compressed blob directly in the state so the render callback doesn't
    /// need to touch the custom parameter again (which would mean re-loading
    /// the whole clip-wide cache for every render on every thread).
    var cachedMatteBlob: Data?
    /// Inference resolution the cached matte was produced at. `0` when no
    /// matte is cached. The render path bypasses the MLX path only when this
    /// matches the quality mode the user is currently requesting.
    var cachedMatteInferenceResolution: Int

    init(
        screenColor: ScreenColor = .green,
        qualityMode: QualityMode = .maximum2048,
        sourcePassthroughEnabled: Bool = true,
        passthroughErodeNormalized: Double = 3.0,
        passthroughBlurNormalized: Double = 7.0,
        alphaBlackPoint: Double = 0.0,
        alphaWhitePoint: Double = 1.0,
        alphaErodeNormalized: Double = 0.0,
        alphaSoftnessNormalized: Double = 0.0,
        alphaGamma: Double = 1.0,
        autoDespeckleEnabled: Bool = false,
        despeckleSize: Int = 100,
        refinerStrength: Double = 1.0,
        despillStrength: Double = 0.5,
        spillMethod: SpillMethod = .average,
        lightWrapEnabled: Bool = false,
        lightWrapStrength: Double = 0.25,
        lightWrapRadius: Double = 10.0,
        edgeDecontaminateEnabled: Bool = false,
        edgeDecontaminateStrength: Double = 0.5,
        temporalStabilityEnabled: Bool = true,
        temporalStabilityStrength: Double = 0.5,
        outputMode: OutputMode = .processed,
        upscaleMethod: UpscaleMethod = .lanczos,
        renderQualityLevel: Int = 2,
        longEdgeBaseline: Double = 1920.0,
        destinationLongEdgePixels: Int = 1920,
        cachedMatteBlob: Data? = nil,
        cachedMatteInferenceResolution: Int = 0
    ) {
        self.screenColor = screenColor
        self.qualityMode = qualityMode
        self.sourcePassthroughEnabled = sourcePassthroughEnabled
        self.passthroughErodeNormalized = passthroughErodeNormalized
        self.passthroughBlurNormalized = passthroughBlurNormalized
        self.alphaBlackPoint = alphaBlackPoint
        self.alphaWhitePoint = alphaWhitePoint
        self.alphaErodeNormalized = alphaErodeNormalized
        self.alphaSoftnessNormalized = alphaSoftnessNormalized
        self.alphaGamma = alphaGamma
        self.autoDespeckleEnabled = autoDespeckleEnabled
        self.despeckleSize = despeckleSize
        self.refinerStrength = refinerStrength
        self.despillStrength = despillStrength
        self.spillMethod = spillMethod
        self.lightWrapEnabled = lightWrapEnabled
        self.lightWrapStrength = lightWrapStrength
        self.lightWrapRadius = lightWrapRadius
        self.edgeDecontaminateEnabled = edgeDecontaminateEnabled
        self.edgeDecontaminateStrength = edgeDecontaminateStrength
        self.temporalStabilityEnabled = temporalStabilityEnabled
        self.temporalStabilityStrength = temporalStabilityStrength
        self.outputMode = outputMode
        self.upscaleMethod = upscaleMethod
        self.renderQualityLevel = renderQualityLevel
        self.longEdgeBaseline = longEdgeBaseline
        self.destinationLongEdgePixels = destinationLongEdgePixels
        self.cachedMatteBlob = cachedMatteBlob
        self.cachedMatteInferenceResolution = cachedMatteInferenceResolution
    }

    // MARK: - Codable

    /// Custom decoding so adding new fields in v1.0 (refiner strength,
    /// light wrap, edge decontamination) doesn't invalidate saved documents
    /// from earlier builds. Missing keys fall back to their default values.
    enum CodingKeys: String, CodingKey {
        case screenColor
        case qualityMode
        case sourcePassthroughEnabled
        case passthroughErodeNormalized
        case passthroughBlurNormalized
        case alphaBlackPoint
        case alphaWhitePoint
        case alphaErodeNormalized
        case alphaSoftnessNormalized
        case alphaGamma
        case autoDespeckleEnabled
        case despeckleSize
        case refinerStrength
        case despillStrength
        case spillMethod
        case lightWrapEnabled
        case lightWrapStrength
        case lightWrapRadius
        case edgeDecontaminateEnabled
        case edgeDecontaminateStrength
        case temporalStabilityEnabled
        case temporalStabilityStrength
        case outputMode
        case upscaleMethod
        case renderQualityLevel
        case longEdgeBaseline
        case destinationLongEdgePixels
        case cachedMatteBlob
        case cachedMatteInferenceResolution
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.screenColor = try container.decodeIfPresent(ScreenColor.self, forKey: .screenColor) ?? .green
        self.qualityMode = try container.decodeIfPresent(QualityMode.self, forKey: .qualityMode) ?? .maximum2048
        self.sourcePassthroughEnabled = try container.decodeIfPresent(Bool.self, forKey: .sourcePassthroughEnabled) ?? true
        self.passthroughErodeNormalized = try container.decodeIfPresent(Double.self, forKey: .passthroughErodeNormalized) ?? 3.0
        self.passthroughBlurNormalized = try container.decodeIfPresent(Double.self, forKey: .passthroughBlurNormalized) ?? 7.0
        self.alphaBlackPoint = try container.decodeIfPresent(Double.self, forKey: .alphaBlackPoint) ?? 0.0
        self.alphaWhitePoint = try container.decodeIfPresent(Double.self, forKey: .alphaWhitePoint) ?? 1.0
        self.alphaErodeNormalized = try container.decodeIfPresent(Double.self, forKey: .alphaErodeNormalized) ?? 0.0
        self.alphaSoftnessNormalized = try container.decodeIfPresent(Double.self, forKey: .alphaSoftnessNormalized) ?? 0.0
        self.alphaGamma = try container.decodeIfPresent(Double.self, forKey: .alphaGamma) ?? 1.0
        self.autoDespeckleEnabled = try container.decodeIfPresent(Bool.self, forKey: .autoDespeckleEnabled) ?? false
        self.despeckleSize = try container.decodeIfPresent(Int.self, forKey: .despeckleSize) ?? 100
        self.refinerStrength = try container.decodeIfPresent(Double.self, forKey: .refinerStrength) ?? 1.0
        self.despillStrength = try container.decodeIfPresent(Double.self, forKey: .despillStrength) ?? 0.5
        self.spillMethod = try container.decodeIfPresent(SpillMethod.self, forKey: .spillMethod) ?? .average
        self.lightWrapEnabled = try container.decodeIfPresent(Bool.self, forKey: .lightWrapEnabled) ?? false
        self.lightWrapStrength = try container.decodeIfPresent(Double.self, forKey: .lightWrapStrength) ?? 0.25
        self.lightWrapRadius = try container.decodeIfPresent(Double.self, forKey: .lightWrapRadius) ?? 10.0
        self.edgeDecontaminateEnabled = try container.decodeIfPresent(Bool.self, forKey: .edgeDecontaminateEnabled) ?? false
        self.edgeDecontaminateStrength = try container.decodeIfPresent(Double.self, forKey: .edgeDecontaminateStrength) ?? 0.5
        self.temporalStabilityEnabled = try container.decodeIfPresent(Bool.self, forKey: .temporalStabilityEnabled) ?? true
        self.temporalStabilityStrength = try container.decodeIfPresent(Double.self, forKey: .temporalStabilityStrength) ?? 0.5
        self.outputMode = try container.decodeIfPresent(OutputMode.self, forKey: .outputMode) ?? .processed
        self.upscaleMethod = try container.decodeIfPresent(UpscaleMethod.self, forKey: .upscaleMethod) ?? .lanczos
        self.renderQualityLevel = try container.decodeIfPresent(Int.self, forKey: .renderQualityLevel) ?? 2
        self.longEdgeBaseline = try container.decodeIfPresent(Double.self, forKey: .longEdgeBaseline) ?? 1920.0
        self.destinationLongEdgePixels = try container.decodeIfPresent(Int.self, forKey: .destinationLongEdgePixels) ?? 1920
        self.cachedMatteBlob = try container.decodeIfPresent(Data.self, forKey: .cachedMatteBlob)
        self.cachedMatteInferenceResolution = try container.decodeIfPresent(Int.self, forKey: .cachedMatteInferenceResolution) ?? 0
    }

    /// Encodes the snapshot for hand-off to the FxPlug host. Binary plist is
    /// used because we need to embed a raw `Data` payload (the cached matte)
    /// efficiently — JSON would base64 it and bloat the blob by ~33 %.
    func encodedForHost() throws -> NSData {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        let data = try encoder.encode(self)
        return NSData(data: data)
    }

    /// Decodes a previously-encoded snapshot. `nil` is tolerated so that the
    /// renderer can fall back to its defaults when the host hands us an empty
    /// blob (for example on plug-in revive).
    static func decoded(from nsData: NSData?) -> PluginStateData {
        guard let nsData, nsData.length > 0 else { return PluginStateData() }
        let data = Data(referencing: nsData)
        do {
            return try PropertyListDecoder().decode(PluginStateData.self, from: data)
        } catch {
            return PluginStateData()
        }
    }

    /// Convenience accessor used by several processors that need to scale a
    /// normalised radius (based on a 1920px long edge) to the current clip.
    func destinationPixelRadius(fromNormalized value: Double) -> Float {
        let scale = Double(destinationLongEdgePixels) / max(longEdgeBaseline, 1.0)
        return Float(value * scale)
    }
}
