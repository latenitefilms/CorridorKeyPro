//
//  CorridorKeyToolboxPlugIn+Parameters.swift
//  CorridorKey by LateNite
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
                userInfo: [NSLocalizedDescriptionKey: "CorridorKey by LateNite could not attach to the FxPlug parameter API."]
            )
        }

        try addHeaderParameter(create: create)
        try addHiddenAnalysisParameter(create: create)
        try addSettingsGroup(create: create)
        try addInteriorDetailGroup(create: create)
        try addMatteGroup(create: create)
        try addEdgeAndSpillGroup(create: create)
        try addEdgeRefinementGroup(create: create)
        try addTemporalStabilityGroup(create: create)
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
        case ParameterIdentifier.analysisData,
             ParameterIdentifier.subjectPoints:
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
        create.addCustomParameter(
            withName: "Subject Points",
            parameterID: ParameterIdentifier.subjectPoints,
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
            defaultValue: UInt32(QualityMode.automatic.rawValue),
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

        // Hint mode defaults to Apple Vision in v1.0 — empirically it
        // beats the green-bias chroma prior on most footage because
        // it segments by subject saliency rather than green-channel
        // dominance. Users with edge-case footage (heavy motion blur,
        // partial occlusion, tricky transparency) can fall back to
        // Automatic or pick Manual to feed only their hint dots.
        //
        // Hint Mode popup replaces the earlier "Auto Subject Hint"
        // toggle with an explicit three-way choice. Sits second in
        // the Settings group (right under Quality) because hint
        // generation is the second-biggest decision driver for the
        // resulting matte.
        create.addPopupMenu(
            withName: "Hint",
            parameterID: ParameterIdentifier.hintMode,
            defaultValue: UInt32(HintMode.appleVision.rawValue),
            menuEntries: HintMode.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        // Subject Marker: a draggable on-screen point that lets the
        // user mark the centre of the foreground subject. Provides
        // both a clear OSC affordance (the user can see where the
        // marker is and drag it) and a stable Point parameter that
        // FCP wires the OSC to natively. Defaults to centre (0.5,
        // 0.5) in object-normalised coords. The Show toggle hides
        // the marker for users who don't want it visible.
        create.addToggleButton(
            withName: "Show Subject Marker",
            parameterID: ParameterIdentifier.showSubjectMarker,
            defaultValue: true,
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )
        create.addPointParameter(
            withName: "Subject Position",
            parameterID: ParameterIdentifier.subjectPosition,
            defaultX: 0.5,
            defaultY: 0.5,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
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
            defaultValue: false,
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

        // Auto Despeckle defaults ON in v1.0 — the connected-components
        // filter sweeps tracking dots, birds, and lens-flare specks
        // out of the background without manual roto. The default
        // 100-pixel size threshold is small enough to leave hair, fur,
        // and feathered edges intact; users who hit edge cases (very
        // wispy hair on noisy plates) can flip it off per clip.
        create.addToggleButton(
            withName: "Auto Despeckle",
            parameterID: ParameterIdentifier.autoDespeckle,
            defaultValue: true,
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

        // Strength stays at 0.5 (the Build 2 default). The method
        // defaults to Ultra (Chroma Project) — its YCbCr-projection
        // path removes the soft chroma fringe the older Average /
        // Double Limit kernels leave on hair and feathered edges,
        // without the over-bright mids Screen Subtract introduces
        // on translucent fabric. Average / Double Limit / Neutral /
        // Screen Subtract remain in the popup for users whose
        // footage responds better to the legacy approaches.
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
            defaultValue: UInt32(SpillMethod.ultra.rawValue),
            menuEntries: SpillMethod.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.endParameterSubGroup()
    }

    /// Phase 4 additions: light wrap and edge colour decontamination.
    /// Both default to enabled in v1.0 so a fresh effect produces a
    /// comp-ready edge — wrap inherits a touch of background colour
    /// to kill the pasted-on look, decontaminate kills the residual
    /// chroma fringe along feathered edges. Saved projects from
    /// builds before v1.0 keep their stored values via
    /// `PluginStateData`'s `decodeIfPresent` defaults so loading an
    /// old library doesn't silently flip the look.
    private func addEdgeRefinementGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Edge Refinement",
            parameterID: ParameterIdentifier.edgeRefinementGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addToggleButton(
            withName: "Light Wrap",
            parameterID: ParameterIdentifier.lightWrapEnabled,
            defaultValue: true,
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
            defaultValue: true,
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

    /// Phase 1 addition: temporal stability. Applied during the Analyse
    /// Clip pass so the cached matte already reflects the blend — zero
    /// hot-path cost at playback. Defaults to enabled so new projects
    /// benefit immediately; saved projects from before this build fall
    /// back to enabled via `PluginStateData.temporalStabilityEnabled`'s
    /// `decodeIfPresent` default, preserving previous output on first
    /// open (there is no cached matte yet, so the first analyse pass
    /// writes a stabilised one).
    private func addTemporalStabilityGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Temporal Stability",
            parameterID: ParameterIdentifier.temporalStabilityGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        // Temporal Stability defaults ON in v1.0 — the motion-gated
        // EMA blend damps the edge-band flicker the model leaves on
        // hair / feather edges. The motion gate keeps fast-moving
        // subjects sharp by skipping the blend wherever source RGB
        // changed more than the gate threshold; users who still see
        // softening can flip it off or pull the strength slider down.
        create.addToggleButton(
            withName: "Reduce Edge Flicker",
            parameterID: ParameterIdentifier.temporalStabilityEnabled,
            defaultValue: true,
            parameterFlags: CorridorKeyParameterFlags.nonAnimatableChoice.fxFlags
        )

        create.addFloatSlider(
            withName: "Stability Strength",
            parameterID: ParameterIdentifier.temporalStabilityStrength,
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

        // Fallback values mirror the v1.0 product defaults — if the
        // retrieval API ever fails, we'd rather assume the slider is
        // live (matching the moef defaults FCP applies to a fresh
        // effect) than greyed-out, which made the inspector look
        // broken when the API briefly stalled.
        let lightWrapOn = readBool(retrieval: retrieval, parameterID: ParameterIdentifier.lightWrapEnabled, at: time, default: true)
        let decontamOn = readBool(retrieval: retrieval, parameterID: ParameterIdentifier.edgeDecontaminateEnabled, at: time, default: true)
        let temporalOn = readBool(retrieval: retrieval, parameterID: ParameterIdentifier.temporalStabilityEnabled, at: time, default: true)

        setEnabled(setter: setter, parameterID: ParameterIdentifier.lightWrapStrength, enabled: lightWrapOn)
        setEnabled(setter: setter, parameterID: ParameterIdentifier.lightWrapRadius, enabled: lightWrapOn)
        setEnabled(setter: setter, parameterID: ParameterIdentifier.edgeDecontaminateStrength, enabled: decontamOn)
        setEnabled(setter: setter, parameterID: ParameterIdentifier.temporalStabilityStrength, enabled: temporalOn)
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
