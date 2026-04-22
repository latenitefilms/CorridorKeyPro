//
//  CorridorKeyProPlugIn+Parameters.swift
//  Corridor Key Pro
//
//  Defines the entire inspector layout for the plug-in. The grouping and
//  defaults intentionally match the CorridorKey OFX panel so that an editor
//  moving between DaVinci Resolve and Final Cut Pro feels at home.
//

import Foundation
import AppKit
import CoreMedia

extension CorridorKeyProPlugIn {

    /// Registers every control FxPlug should draw in Final Cut Pro's inspector.
    /// Called once by FxPlug for each new instance.
    @objc(addParametersWithError:)
    func addParameters() throws {
        guard let create = apiManager.api(for: (any FxParameterCreationAPI_v5).self) as? any FxParameterCreationAPI_v5 else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_APIUnavailable,
                userInfo: [NSLocalizedDescriptionKey: "Corridor Key Pro could not attach to the FxPlug parameter API."]
            )
        }

        try addKeySetupGroup(create: create)
        try addInteriorDetailGroup(create: create)
        try addMatteGroup(create: create)
        try addEdgeAndSpillGroup(create: create)
        try addOutputGroup(create: create)
        try addPerformanceGroup(create: create)
        try addProcessGroup(create: create)
        PluginLog.notice("Parameters registered with Final Cut Pro.")
    }

    // MARK: - Groups

    private func addKeySetupGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Key Setup",
            parameterID: ParameterIdentifier.keySetupGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Screen Colour",
            parameterID: ParameterIdentifier.screenColor,
            defaultValue: UInt32(ScreenColor.green.rawValue),
            menuEntries: ScreenColor.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Quality",
            parameterID: ParameterIdentifier.qualityMode,
            defaultValue: UInt32(QualityMode.automatic.rawValue),
            menuEntries: QualityMode.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addImageReference(
            withName: "Alpha Hint",
            parameterID: ParameterIdentifier.alphaHintClip,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addInteriorDetailGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Interior Detail",
            parameterID: ParameterIdentifier.interiorGroup,
            parameterFlags: [CorridorKeyParameterFlags.default, .collapsed].reduce(into: CorridorKeyParameterFlags()) { $0.insert($1) }.fxFlags
        )

        create.addToggleButton(
            withName: "Source Passthrough",
            parameterID: ParameterIdentifier.sourcePassthrough,
            defaultValue: true,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
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
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addIntSlider(
            withName: "Despeckle Size",
            parameterID: ParameterIdentifier.despeckleSize,
            defaultValue: 400,
            parameterMin: 50,
            parameterMax: 2000,
            sliderMin: 50,
            sliderMax: 2000,
            delta: 10,
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
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addOutputGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Output",
            parameterID: ParameterIdentifier.outputGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Output",
            parameterID: ParameterIdentifier.outputMode,
            defaultValue: UInt32(OutputMode.processed.rawValue),
            menuEntries: OutputMode.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addPerformanceGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Performance",
            parameterID: ParameterIdentifier.performanceGroup,
            parameterFlags: [CorridorKeyParameterFlags.default, .collapsed].reduce(into: CorridorKeyParameterFlags()) { $0.insert($1) }.fxFlags
        )

        create.addFloatSlider(
            withName: "Temporal Smoothing",
            parameterID: ParameterIdentifier.temporalSmoothing,
            defaultValue: 0,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Upscale Method",
            parameterID: ParameterIdentifier.upscaleMethod,
            defaultValue: UInt32(UpscaleMethod.bilinear.rawValue),
            menuEntries: UpscaleMethod.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addProcessGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Process",
            parameterID: ParameterIdentifier.processGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPushButton(
            withName: "Process Clip",
            parameterID: ParameterIdentifier.processClipButton,
            selector: #selector(handleProcessClip),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPushButton(
            withName: "Open User Guide",
            parameterID: ParameterIdentifier.openUserGuide,
            selector: #selector(handleOpenUserGuide),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    // MARK: - Callbacks

    @objc func handleProcessClip() {
        PluginLog.notice("Process Clip button pressed.")
        guard let analysisObject = apiManager.api(for: (any FxAnalysisAPI_v2).self)
                ?? apiManager.api(for: (any FxAnalysisAPI).self) else {
            PluginLog.error("Process Clip: FxAnalysisAPI is not available on this host.")
            presentProcessAlert(
                title: "Process Clip unavailable",
                message: "This host does not expose the analysis API. Live rendering will continue using the rough matte."
            )
            return
        }

        do {
            if let analysis = analysisObject as? any FxAnalysisAPI_v2 {
                try analysis.startForwardAnalysis(kFxAnalysisLocation_GPU)
            } else if let analysis = analysisObject as? any FxAnalysisAPI {
                try analysis.startForwardAnalysis(kFxAnalysisLocation_GPU)
            } else {
                throw NSError(domain: FxPlugErrorDomain, code: kFxError_APIUnavailable)
            }
            PluginLog.notice("Forward analysis started on GPU.")
            presentProcessAlert(
                title: "Processing started",
                message: "Final Cut Pro is analysing the clip in the background. Progress is shown in the timeline overlay."
            )
        } catch {
            PluginLog.error("startForwardAnalysis failed: \(error.localizedDescription)")
            presentProcessAlert(
                title: "Processing could not start",
                message: error.localizedDescription
            )
        }
    }

    @objc func handleOpenUserGuide() {
        guard let url = URL(string: "https://corridordigital.com/corridor-key-pro") else { return }
        NSWorkspace.shared.open(url)
    }

    /// Shows a short confirmation alert over the FCP UI. Posted through the
    /// main queue because the button handler arrives on an arbitrary thread.
    private func presentProcessAlert(title: String, message: String) {
        Task { @MainActor in
            let alert = NSAlert()
            alert.messageText = title
            alert.informativeText = message
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }

    // MARK: - Parameter change notifications

    /// Final Cut Pro invokes this method when the user edits a control. We
    /// don't need to react per-parameter — the host will re-request the
    /// plug-in state and re-render automatically — but having it defined
    /// ensures FCP correctly invalidates any cached render when a control
    /// changes.
    @objc(parameterChanged:atTime:error:)
    func parameterChanged(_ paramID: UInt32, atTime time: CMTime) throws {
        PluginLog.debug("Parameter \(paramID) changed at \(CMTimeGetSeconds(time))s.")
    }
}
