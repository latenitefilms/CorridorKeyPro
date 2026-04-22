//
//  ParameterIdentifiers.swift
//  Corridor Key Toolbox
//
//  Every parameter surfaced to Final Cut Pro has a stable numeric identifier.
//  FxPlug requires identifiers to stay in the range 1...9998 for the lifetime
//  of the plug-in — renaming is fine but renumbering will strand user
//  documents, so new parameters must always be appended with a fresh id.
//

import Foundation

enum ParameterIdentifier {
    // Subgroups
    static let settingsGroup: UInt32 = 100
    static let interiorGroup: UInt32 = 110
    static let matteGroup: UInt32 = 120
    static let edgeSpillGroup: UInt32 = 130

    /// Custom-UI parameter that hosts the inspector header (app icon,
    /// version, Analyse/Reset buttons, analysis status). Drawn as a SwiftUI
    /// `NSHostingView` returned from `createViewForParameterID`.
    static let headerSummary: UInt32 = 50

    // Settings
    static let screenColor: UInt32 = 1001
    static let qualityMode: UInt32 = 1002

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

    // Output (lives in the Settings group)
    static let outputMode: UInt32 = 5001

    // Performance (also in the Settings group)
    static let upscaleMethod: UInt32 = 6002

    /// Hidden custom parameter that persists the per-frame MLX mattes inside
    /// the Final Cut Pro Library so editors can move projects between
    /// machines without losing the analysed cache.
    static let analysisData: UInt32 = 7003
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
    static let customUI = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_CUSTOM_UI))
    static let useFullViewWidth = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_USE_FULL_VIEW_WIDTH))
    static let curveEditorHidden = CorridorKeyParameterFlags(rawValue: UInt32(kFxParameterFlag_CURVE_EDITOR_HIDDEN))

    /// Flag combo we apply to every enum/toggle parameter in the inspector.
    /// Animating screen colour or output mode makes no visual sense and
    /// clutters the curve editor, so we keep these controls static.
    static let nonAnimatableChoice: CorridorKeyParameterFlags = [.default, .notAnimatable, .curveEditorHidden]

    /// Flag combo for the custom-UI inspector header parameter.
    static let headerCustomUI: CorridorKeyParameterFlags = [
        .default,
        .customUI,
        .notAnimatable,
        .curveEditorHidden,
        .useFullViewWidth
    ]

    var fxFlags: FxParameterFlags { FxParameterFlags(rawValue) }
}
