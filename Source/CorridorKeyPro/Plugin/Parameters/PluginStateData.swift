//
//  PluginStateData.swift
//  Corridor Key Pro
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

    // Edge and spill
    var despillStrength: Double
    var spillMethod: SpillMethod

    // Output
    var outputMode: OutputMode

    // Performance and quality
    var temporalSmoothing: Double
    var upscaleMethod: UpscaleMethod

    // Runtime
    var renderQualityLevel: Int

    /// Long edge used by the parameter panel to scale pixel-space controls to
    /// the current clip. Mirrors the OFX plugin's 1920px baseline behaviour.
    var longEdgeBaseline: Double

    /// Actual long edge of the destination image at render time, used to
    /// translate normalised radii into destination pixels.
    var destinationLongEdgePixels: Int

    init(
        screenColor: ScreenColor = .green,
        qualityMode: QualityMode = .draft512,
        sourcePassthroughEnabled: Bool = true,
        passthroughErodeNormalized: Double = 3.0,
        passthroughBlurNormalized: Double = 7.0,
        alphaBlackPoint: Double = 0.0,
        alphaWhitePoint: Double = 1.0,
        alphaErodeNormalized: Double = 0.0,
        alphaSoftnessNormalized: Double = 0.0,
        alphaGamma: Double = 1.0,
        autoDespeckleEnabled: Bool = false,
        despeckleSize: Int = 400,
        despillStrength: Double = 0.5,
        spillMethod: SpillMethod = .average,
        outputMode: OutputMode = .processed,
        temporalSmoothing: Double = 0.0,
        upscaleMethod: UpscaleMethod = .bilinear,
        renderQualityLevel: Int = 2,
        longEdgeBaseline: Double = 1920.0,
        destinationLongEdgePixels: Int = 1920
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
        self.despillStrength = despillStrength
        self.spillMethod = spillMethod
        self.outputMode = outputMode
        self.temporalSmoothing = temporalSmoothing
        self.upscaleMethod = upscaleMethod
        self.renderQualityLevel = renderQualityLevel
        self.longEdgeBaseline = longEdgeBaseline
        self.destinationLongEdgePixels = destinationLongEdgePixels
    }

    /// Encodes the snapshot as NSData for hand-off to the FxPlug host. A dedicated
    /// encoder keeps the format stable across processes.
    func encodedForHost() throws -> NSData {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
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
            return try JSONDecoder().decode(PluginStateData.self, from: data)
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
