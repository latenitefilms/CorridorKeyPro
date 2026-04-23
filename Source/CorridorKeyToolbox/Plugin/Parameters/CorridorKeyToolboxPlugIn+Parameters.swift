//
//  CorridorKeyToolboxPlugIn+Parameters.swift
//  Corridor Key Toolbox
//
//  Defines the entire inspector layout for the plug-in. Analyse / Reset /
//  version live in the custom-UI header at the top, then the sliders match
//  the CorridorKey OFX panel so editors moving between Resolve and Final
//  Cut Pro stay oriented. Enum and toggle controls are explicitly marked
//  non-animatable — animating "Screen Colour" or "Output" flashes between
//  unrelated looks and pollutes the curve editor.
//

import Foundation
import AppKit
import CoreMedia

extension CorridorKeyToolboxPlugIn {

    /// Registers every control FxPlug should draw in Final Cut Pro's inspector.
    /// Called once by FxPlug for each new instance.
    @objc(addParametersWithError:)
    func addParameters() throws {
        guard let create = apiManager.api(for: (any FxParameterCreationAPI_v5).self) as? any FxParameterCreationAPI_v5 else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_APIUnavailable,
                userInfo: [NSLocalizedDescriptionKey: "Corridor Key Toolbox could not attach to the FxPlug parameter API."]
            )
        }

        try addHeaderParameter(create: create)
        try addHiddenAnalysisParameter(create: create)
        try addSettingsGroup(create: create)
        try addInteriorDetailGroup(create: create)
        try addMatteGroup(create: create)
        try addEdgeAndSpillGroup(create: create)
        try addEdgeRefinementGroup(create: create)
        PluginLog.notice("Parameters registered with Final Cut Pro.")
    }

    // MARK: - Custom parameter serialisation

    /// Tells FxPlug which Foundation classes may appear inside our hidden
    /// custom parameter so the host can safely decode it through
    /// `NSSecureCoding`. The analysis dictionary is a tree of NSDictionary /
    /// NSString / NSNumber / NSData nodes — listing any additional classes
    /// here would open the plug-in to decoding attacker-supplied payloads.
    func classes(forCustomParameterID parameterID: UInt32) -> Set<AnyHashable> {
        switch parameterID {
        case ParameterIdentifier.analysisData:
            let classes: [AnyClass] = [
                NSDictionary.self,
                NSString.self,
                NSNumber.self,
                NSData.self
            ]
            let set = NSSet(array: classes)
            return (set as? Set<AnyHashable>) ?? []
        default:
            return []
        }
    }

    // MARK: - Inspector header + hidden cache

    /// Custom-UI placeholder that `createViewForParameterID` swaps for the
    /// SwiftUI header (icon, version, Analyse / Reset buttons, status).
    private func addHeaderParameter(create: any FxParameterCreationAPI_v5) throws {
        create.addCustomParameter(
            withName: "",
            parameterID: ParameterIdentifier.headerSummary,
            defaultValue: NSDictionary(),
            parameterFlags: CorridorKeyParameterFlags.headerCustomUI.fxFlags
        )
    }

    /// Hidden, non-animatable custom parameter that persists the per-frame
    /// MLX mattes inside the Final Cut Pro Library. Never surfaced to the
    /// inspector — only touched by the analyser and by `pluginState`.
    private func addHiddenAnalysisParameter(create: any FxParameterCreationAPI_v5) throws {
        let hiddenFlags: CorridorKeyParameterFlags = [.default, .hidden, .notAnimatable, .curveEditorHidden]
        create.addCustomParameter(
            withName: "Analysis Data",
            parameterID: ParameterIdentifier.analysisData,
            defaultValue: NSDictionary(),
            parameterFlags: hiddenFlags.fxFlags
        )
    }

    // MARK: - Visible groups

    private func addSettingsGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Settings",
            parameterID: ParameterIdentifier.settingsGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Quality",
            parameterID: ParameterIdentifier.qualityMode,
            defaultValue: UInt32(QualityMode.maximum2048.rawValue),
            menuEntries: QualityMode.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addPopupMenu(
            withName: "Screen Colour",
            parameterID: ParameterIdentifier.screenColor,
            defaultValue: UInt32(ScreenColor.green.rawValue),
            menuEntries: ScreenColor.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addPopupMenu(
            withName: "Upscale Method",
            parameterID: ParameterIdentifier.upscaleMethod,
            defaultValue: UInt32(UpscaleMethod.lanczos.rawValue),
            menuEntries: UpscaleMethod.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addPopupMenu(
            withName: "Output",
            parameterID: ParameterIdentifier.outputMode,
            defaultValue: UInt32(OutputMode.processed.rawValue),
            menuEntries: OutputMode.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )
        
        create.endParameterSubGroup()
    }

    private func addInteriorDetailGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Interior Detail",
            parameterID: ParameterIdentifier.interiorGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addToggleButton(
            withName: "Source Passthrough",
            parameterID: ParameterIdentifier.sourcePassthrough,
            defaultValue: true,
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addFloatSlider(
            withName: "Edge Erode",
            parameterID: ParameterIdentifier.passthroughErode,
            defaultValue: 3,
            parameterMin: 0,
            parameterMax: 100,
            sliderMin: 0,
            sliderMax: 40,
            delta: 1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Edge Blur",
            parameterID: ParameterIdentifier.passthroughBlur,
            defaultValue: 7,
            parameterMin: 0,
            parameterMax: 100,
            sliderMin: 0,
            sliderMax: 40,
            delta: 1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addMatteGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Matte",
            parameterID: ParameterIdentifier.matteGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Black Point",
            parameterID: ParameterIdentifier.alphaBlackPoint,
            defaultValue: 0,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "White Point",
            parameterID: ParameterIdentifier.alphaWhitePoint,
            defaultValue: 1,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Matte Erode",
            parameterID: ParameterIdentifier.alphaErode,
            defaultValue: 0,
            parameterMin: -10,
            parameterMax: 10,
            sliderMin: -10,
            sliderMax: 10,
            delta: 0.1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Softness",
            parameterID: ParameterIdentifier.alphaSoftness,
            defaultValue: 0,
            parameterMin: 0,
            parameterMax: 5,
            sliderMin: 0,
            sliderMax: 5,
            delta: 0.1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Gamma",
            parameterID: ParameterIdentifier.alphaGamma,
            defaultValue: 1,
            parameterMin: 0.1,
            parameterMax: 10,
            sliderMin: 0.1,
            sliderMax: 4,
            delta: 0.05,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addToggleButton(
            withName: "Auto Despeckle",
            parameterID: ParameterIdentifier.autoDespeckle,
            defaultValue: false,
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addIntSlider(
            withName: "Despeckle Size",
            parameterID: ParameterIdentifier.despeckleSize,
            defaultValue: 100,
            parameterMin: 5,
            parameterMax: 2000,
            sliderMin: 5,
            sliderMax: 1000,
            delta: 5,
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addFloatSlider(
            withName: "Refiner Strength",
            parameterID: ParameterIdentifier.refinerStrength,
            defaultValue: 1.0,
            parameterMin: 0,
            parameterMax: 2,
            sliderMin: 0,
            sliderMax: 2,
            delta: 0.05,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addEdgeAndSpillGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Edge & Spill",
            parameterID: ParameterIdentifier.edgeSpillGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Despill Strength",
            parameterID: ParameterIdentifier.despillStrength,
            defaultValue: 0.5,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Spill Method",
            parameterID: ParameterIdentifier.spillMethod,
            defaultValue: UInt32(SpillMethod.average.rawValue),
            menuEntries: SpillMethod.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.endParameterSubGroup()
    }

    /// Phase 4 additions: light wrap and edge colour decontamination. Both
    /// default to disabled so existing projects render the same on upgrade.
    private func addEdgeRefinementGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Edge Refinement",
            parameterID: ParameterIdentifier.edgeRefinementGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addToggleButton(
            withName: "Light Wrap",
            parameterID: ParameterIdentifier.lightWrapEnabled,
            defaultValue: false,
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addFloatSlider(
            withName: "Wrap Strength",
            parameterID: ParameterIdentifier.lightWrapStrength,
            defaultValue: 0.25,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Wrap Radius",
            parameterID: ParameterIdentifier.lightWrapRadius,
            defaultValue: 10,
            parameterMin: 0,
            parameterMax: 50,
            sliderMin: 0,
            sliderMax: 50,
            delta: 0.5,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addToggleButton(
            withName: "Edge Decontaminate",
            parameterID: ParameterIdentifier.edgeDecontaminateEnabled,
            defaultValue: false,
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addFloatSlider(
            withName: "Decontam. Strength",
            parameterID: ParameterIdentifier.edgeDecontaminateStrength,
            defaultValue: 0.5,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    // MARK: - Parameter change notifications

    /// Final Cut Pro invokes this method when the user edits a control. We
    /// use it to keep the inspector UI consistent — when the user toggles
    /// Light Wrap or Edge Decontaminate off, the sub-sliders get disabled
    /// so the panel clearly communicates which knobs are live.
    @objc(parameterChanged:atTime:error:)
    func parameterChanged(_ paramID: UInt32, atTime time: CMTime) throws {
        updateDependentParameterEnablement(at: time)
    }

    /// Reads each Phase 4 toggle and disables / enables its sub-sliders
    /// accordingly. Safe to call any time; no-ops when the parameter APIs
    /// aren't available (e.g. during initial add-parameters).
    private func updateDependentParameterEnablement(at time: CMTime) {
        guard
            let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6,
            let setter = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5
        else { return }

        let lightWrapOn = readBool(retrieval: retrieval, parameterID: ParameterIdentifier.lightWrapEnabled, at: time, default: false)
        let decontamOn = readBool(retrieval: retrieval, parameterID: ParameterIdentifier.edgeDecontaminateEnabled, at: time, default: false)

        setEnabled(setter: setter, parameterID: ParameterIdentifier.lightWrapStrength, enabled: lightWrapOn)
        setEnabled(setter: setter, parameterID: ParameterIdentifier.lightWrapRadius, enabled: lightWrapOn)
        setEnabled(setter: setter, parameterID: ParameterIdentifier.edgeDecontaminateStrength, enabled: decontamOn)
    }

    private func readBool(
        retrieval: any FxParameterRetrievalAPI_v6,
        parameterID: UInt32,
        at time: CMTime,
        default defaultValue: Bool
    ) -> Bool {
        var value = ObjCBool(defaultValue)
        retrieval.getBoolValue(&value, fromParameter: parameterID, at: time)
        return value.boolValue
    }

    private func setEnabled(
        setter: any FxParameterSettingAPI_v5,
        parameterID: UInt32,
        enabled: Bool
    ) {
        let flags: CorridorKeyParameterFlags = enabled ? .default : [.default, .disabled]
        _ = setter.setParameterFlags(flags.fxFlags, toParameter: parameterID)
    }
}
