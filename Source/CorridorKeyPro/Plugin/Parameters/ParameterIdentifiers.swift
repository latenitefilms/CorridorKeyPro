//
//  ParameterIdentifiers.swift
//  Corridor Key Pro
//
//  Every parameter surfaced to Final Cut Pro has a stable numeric identifier.
//  FxPlug requires identifiers to stay in the range 1...9998 for the lifetime
//  of the plug-in — renaming is fine but renumbering will strand user
//  documents, so new parameters must always be appended with a fresh id.
//

import Foundation

enum ParameterIdentifier {
    // Subgroups
    static let keySetupGroup: UInt32 = 100
    static let interiorGroup: UInt32 = 110
    static let matteGroup: UInt32 = 120
    static let edgeSpillGroup: UInt32 = 130
    static let outputGroup: UInt32 = 140
    static let performanceGroup: UInt32 = 150
    static let advancedGroup: UInt32 = 160
    static let runtimeStatusGroup: UInt32 = 170

    // Key Setup
    static let screenColor: UInt32 = 1001
    static let qualityMode: UInt32 = 1002
    static let inputColorSpace: UInt32 = 1003
    static let alphaHintClip: UInt32 = 1004

    // Interior Detail
    static let sourcePassthrough: UInt32 = 2001
    static let passthroughErode: UInt32 = 2002
    static let passthroughBlur: UInt32 = 2003

    // Matte refinement
    static let alphaBlackPoint: UInt32 = 3001
    static let alphaWhitePoint: UInt32 = 3002
    static let alphaErode: UInt32 = 3003
    static let alphaSoftness: UInt32 = 3004
    static let alphaGamma: UInt32 = 3005
    static let autoDespeckle: UInt32 = 3006
    static let despeckleSize: UInt32 = 3007

    // Edge and spill
    static let despillStrength: UInt32 = 4001
    static let spillMethod: UInt32 = 4002

    // Output
    static let outputMode: UInt32 = 5001

    // Performance
    static let temporalSmoothing: UInt32 = 6001
    static let upscaleMethod: UInt32 = 6002

    // Advanced runtime
    static let allowCPUFallback: UInt32 = 7001
    static let renderTimeoutSeconds: UInt32 = 7002

    // Read-only runtime status (string parameters the render step updates).
    static let statusBackend: UInt32 = 8001
    static let statusEffectiveQuality: UInt32 = 8002
    static let statusGuideSource: UInt32 = 8003
    static let statusLastFrameMs: UInt32 = 8004
    static let statusDevice: UInt32 = 8005

    // Push buttons
    static let openUserGuide: UInt32 = 9001
}

/// Convenience wrapper that makes the raw `kFxParameterFlag_*` constants feel at
/// home in modern Swift. Flags are combined with bitwise OR just like their C
/// counterparts.
struct CorridorKeyParameterFlags: OptionSet, Sendable {
    let rawValue: UInt32
    init(rawValue: UInt32) { self.rawValue = rawValue }

    static let `default` = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_DEFAULT))
    static let hidden = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_HIDDEN))
    static let disabled = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_DISABLED))
    static let notAnimatable = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_NOT_ANIMATABLE))
    static let collapsed = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_COLLAPSED))
    static let ignoreMinMax = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_IGNORE_MINMAX))

    var fxFlags: FxParameterFlags { FxParameterFlags(rawValue) }
}
