//
//  ParameterRanges.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Single source of truth for slider min / max / default / step values.
//  Mirrors the FxPlug parameter table in
//  `CorridorKeyToolboxPlugIn+Parameters.swift` so the editor's
//  inspector behaves identically to Final Cut Pro's.
//
//  Centralising the metadata here means adding a new control or
//  tweaking a range only happens in one place per host.
//

import Foundation

/// Slider metadata. Mirrors `addFloatSlider` / `addIntSlider`
/// arguments on the FxPlug side.
struct FloatParameterRange: Sendable {
    let name: String
    let defaultValue: Double
    let parameterMin: Double
    let parameterMax: Double
    let sliderMin: Double
    let sliderMax: Double
    let step: Double
}

struct IntParameterRange: Sendable {
    let name: String
    let defaultValue: Int
    let parameterMin: Int
    let parameterMax: Int
    let sliderMin: Int
    let sliderMax: Int
    let step: Int
}

enum ParameterRanges {

    // Settings
    static let qualityModeName = "Quality"
    static let hintModeName = "Hint"
    static let screenColorName = "Screen Colour"
    static let upscaleMethodName = "Upscale Method"
    static let outputModeName = "Output"

    // Interior detail
    static let sourcePassthroughName = "Source Passthrough"
    static let passthroughErode = FloatParameterRange(
        name: "Edge Erode",
        defaultValue: 3, parameterMin: 0, parameterMax: 100,
        sliderMin: 0, sliderMax: 40, step: 1
    )
    static let passthroughBlur = FloatParameterRange(
        name: "Edge Blur",
        defaultValue: 7, parameterMin: 0, parameterMax: 100,
        sliderMin: 0, sliderMax: 40, step: 1
    )

    // Matte
    static let alphaBlackPoint = FloatParameterRange(
        name: "Black Point",
        defaultValue: 0, parameterMin: 0, parameterMax: 1,
        sliderMin: 0, sliderMax: 1, step: 0.01
    )
    static let alphaWhitePoint = FloatParameterRange(
        name: "White Point",
        defaultValue: 1, parameterMin: 0, parameterMax: 1,
        sliderMin: 0, sliderMax: 1, step: 0.01
    )
    static let alphaErode = FloatParameterRange(
        name: "Matte Erode",
        defaultValue: 0, parameterMin: -10, parameterMax: 10,
        sliderMin: -10, sliderMax: 10, step: 0.1
    )
    static let alphaSoftness = FloatParameterRange(
        name: "Softness",
        defaultValue: 0, parameterMin: 0, parameterMax: 5,
        sliderMin: 0, sliderMax: 5, step: 0.1
    )
    static let alphaGamma = FloatParameterRange(
        name: "Gamma",
        defaultValue: 1, parameterMin: 0.1, parameterMax: 10,
        sliderMin: 0.1, sliderMax: 4, step: 0.05
    )
    static let autoDespeckleName = "Auto Despeckle"
    static let despeckleSize = IntParameterRange(
        name: "Despeckle Size",
        defaultValue: 100, parameterMin: 5, parameterMax: 2000,
        sliderMin: 5, sliderMax: 1000, step: 5
    )
    static let refinerStrength = FloatParameterRange(
        name: "Refiner Strength",
        defaultValue: 1.0, parameterMin: 0, parameterMax: 2,
        sliderMin: 0, sliderMax: 2, step: 0.05
    )

    // Edge & spill
    static let despillStrength = FloatParameterRange(
        name: "Despill Strength",
        defaultValue: 0.5, parameterMin: 0, parameterMax: 1,
        sliderMin: 0, sliderMax: 1, step: 0.01
    )
    static let spillMethodName = "Spill Method"

    // Edge refinement
    static let lightWrapName = "Light Wrap"
    static let lightWrapStrength = FloatParameterRange(
        name: "Wrap Strength",
        defaultValue: 0.25, parameterMin: 0, parameterMax: 1,
        sliderMin: 0, sliderMax: 1, step: 0.01
    )
    static let lightWrapRadius = FloatParameterRange(
        name: "Wrap Radius",
        defaultValue: 10, parameterMin: 0, parameterMax: 50,
        sliderMin: 0, sliderMax: 50, step: 0.5
    )
    static let edgeDecontaminateName = "Edge Decontaminate"
    static let edgeDecontaminateStrength = FloatParameterRange(
        name: "Decontam. Strength",
        defaultValue: 0.5, parameterMin: 0, parameterMax: 1,
        sliderMin: 0, sliderMax: 1, step: 0.01
    )

    // Temporal stability
    static let temporalStabilityName = "Reduce Edge Flicker"
    static let temporalStabilityStrength = FloatParameterRange(
        name: "Stability Strength",
        defaultValue: 0.5, parameterMin: 0, parameterMax: 1,
        sliderMin: 0, sliderMax: 1, step: 0.01
    )
}
