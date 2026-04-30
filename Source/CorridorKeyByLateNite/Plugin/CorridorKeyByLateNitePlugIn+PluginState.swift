//
//  CorridorKeyToolboxPlugIn+PluginState.swift
//  CorridorKey by LateNite
//
//  Reads every parameter into a single value-type snapshot that the renderer
//  and analyser can consume without touching the FxPlug API again. FxPlug
//  requires us to return a fresh NSData each call because the host may issue
//  calls from multiple threads.
//

import Foundation
import CoreMedia

enum HeaderCustomParameterStorage {
    static func dictionaryValue(
        for key: String,
        using retrieval: any FxParameterRetrievalAPI_v6
    ) -> NSDictionary? {
        let storage = storageDictionary(using: retrieval)
        if let dictionary = storage[NSString(string: key)] as? NSDictionary {
            return dictionary
        }
        return storage[key] as? NSDictionary
    }

    @discardableResult
    static func setDictionary(
        _ dictionary: NSDictionary,
        for key: String,
        using apiManager: any PROAPIAccessing
    ) -> Bool {
        guard let settingAPI = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            return false
        }

        let storage = NSMutableDictionary()
        if let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 {
            for (existingKey, existingValue) in storageDictionary(using: retrieval) {
                storage[existingKey] = existingValue
            }
        }
        storage[NSString(string: HeaderStorageKey.schemaVersion)] = NSNumber(value: 1)
        storage[NSString(string: key)] = dictionary

        let value = storage.copy() as? NSDictionary ?? NSDictionary()
        return settingAPI.setCustomParameterValue(
            value,
            toParameter: ParameterIdentifier.headerSummary,
            at: CMTime.zero
        )
    }

    private static func storageDictionary(using retrieval: any FxParameterRetrievalAPI_v6) -> NSDictionary {
        var rawValue: (any NSCopying & NSObjectProtocol & NSSecureCoding)?
        retrieval.getCustomParameterValue(
            &rawValue,
            fromParameter: ParameterIdentifier.headerSummary,
            at: CMTime.zero
        )
        return rawValue as? NSDictionary ?? NSDictionary()
    }
}

extension CorridorKeyToolboxPlugIn {

    @objc(pluginState:atTime:quality:error:)
    func pluginState(
        _ pluginState: AutoreleasingUnsafeMutablePointer<NSData>?,
        at renderTime: CMTime,
        quality qualityLevel: UInt
    ) throws {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_APIUnavailable,
                userInfo: [NSLocalizedDescriptionKey: "CorridorKey by LateNite could not read parameter values."]
            )
        }

        var state = PluginStateData()
        state.renderQualityLevel = Int(qualityLevel)

        // Key setup
        state.screenColor = popupValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.screenColor,
            time: renderTime,
            default: .green
        )
        state.qualityMode = popupValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.qualityMode,
            time: renderTime,
            default: .automatic
        )
        // Read the new Hint Mode popup. The popup's default is
        // `.appleVision`, matching what the legacy "Auto Subject
        // Hint = on" toggle did, so projects opening for the first
        // time after this change keep their hint behaviour.
        state.hintMode = popupValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.hintMode,
            time: renderTime,
            default: HintMode.appleVision
        )
        state.showSubjectMarker = boolValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.showSubjectMarker,
            time: renderTime,
            default: true
        )
        let (subjectX, subjectY) = pointValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.subjectPosition,
            time: renderTime,
            defaultX: 0.5,
            defaultY: 0.5
        )
        state.subjectPositionX = subjectX
        state.subjectPositionY = subjectY

        // Interior. Defaults below mirror `PluginStateData()`'s
        // designated initialiser exactly so the FxPlug and the
        // Standalone Editor land on the same matte when nothing
        // overrides — earlier drift here meant a fresh FCP project
        // ran the keyer with Source Passthrough on / Auto Despeckle
        // off / Light Wrap off / Edge Decontaminate off / Temporal
        // Stability off, while the Standalone Editor ran with the
        // opposite combination.
        state.sourcePassthroughEnabled = boolValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.sourcePassthrough,
            time: renderTime,
            default: false
        )
        state.passthroughErodeNormalized = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.passthroughErode,
            time: renderTime,
            default: 3
        )
        state.passthroughBlurNormalized = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.passthroughBlur,
            time: renderTime,
            default: 7
        )

        // Matte
        state.alphaBlackPoint = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.alphaBlackPoint,
            time: renderTime,
            default: 0
        )
        state.alphaWhitePoint = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.alphaWhitePoint,
            time: renderTime,
            default: 1
        )
        state.alphaErodeNormalized = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.alphaErode,
            time: renderTime,
            default: 0
        )
        state.alphaSoftnessNormalized = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.alphaSoftness,
            time: renderTime,
            default: 0
        )
        state.alphaGamma = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.alphaGamma,
            time: renderTime,
            default: 1
        )
        state.autoDespeckleEnabled = boolValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.autoDespeckle,
            time: renderTime,
            default: true
        )
        state.despeckleSize = intValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.despeckleSize,
            time: renderTime,
            default: 100
        )
        state.refinerStrength = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.refinerStrength,
            time: renderTime,
            default: 1.0
        )

        // Edge refinement (Phase 4).
        state.lightWrapEnabled = boolValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.lightWrapEnabled,
            time: renderTime,
            default: true
        )
        state.lightWrapStrength = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.lightWrapStrength,
            time: renderTime,
            default: 0.25
        )
        state.lightWrapRadius = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.lightWrapRadius,
            time: renderTime,
            default: 10
        )
        state.edgeDecontaminateEnabled = boolValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.edgeDecontaminateEnabled,
            time: renderTime,
            default: true
        )
        state.edgeDecontaminateStrength = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.edgeDecontaminateStrength,
            time: renderTime,
            default: 0.5
        )

        // Temporal stability (Phase 1).
        state.temporalStabilityEnabled = boolValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.temporalStabilityEnabled,
            time: renderTime,
            default: true
        )
        state.temporalStabilityStrength = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.temporalStabilityStrength,
            time: renderTime,
            default: 0.5
        )

        // Edge & spill
        state.despillStrength = floatValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.despillStrength,
            time: renderTime,
            default: 0.5
        )
        state.spillMethod = popupValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.spillMethod,
            time: renderTime,
            default: .ultra
        )

        // Output and performance
        state.outputMode = popupValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.outputMode,
            time: renderTime,
            default: .processed
        )
        state.upscaleMethod = popupValue(
            retrieval: retrieval,
            parameterID: ParameterIdentifier.upscaleMethod,
            time: renderTime,
            default: .lanczos
        )

        // If FxAnalyzer has cached a matte for this frame, pack the
        // compressed blob and its inference resolution into the state so the
        // render callback can skip MLX entirely. Screen colour must match the
        // analysis run — mismatches invalidate the cache for that frame.
        if let analysis = loadAnalysisData(using: retrieval),
           let cachedMatte = analysis.cachedMatte(
                at: renderTime,
                screenColorRaw: state.screenColor.rawValue
           ) {
            state.cachedMatteBlob = cachedMatte.blob
            state.cachedMatteInferenceResolution = cachedMatte.inferenceResolution
        }

        // Load the user's OSC-placed hint dots from the published header
        // storage dictionary so they round-trip through the `.moef` wrapper.
        state.hintPointSet = loadHintPointSet(using: retrieval)

        let nsData = try state.encodedForHost()
        pluginState?.pointee = nsData
    }

    // MARK: - Retrieval helpers

    /// Reads a choice parameter, falling back to the supplied default if the
    /// host signals an error or returns an index outside of the known range.
    private func popupValue<Enum: RawRepresentable>(
        retrieval: any FxParameterRetrievalAPI_v6,
        parameterID: UInt32,
        time: CMTime,
        default defaultValue: Enum
    ) -> Enum where Enum.RawValue == Int {
        var raw: Int32 = 0
        guard retrieval.getIntValue(&raw, fromParameter: parameterID, at: time) else {
            return defaultValue
        }
        return Enum(rawValue: Int(raw)) ?? defaultValue
    }

    private func floatValue(
        retrieval: any FxParameterRetrievalAPI_v6,
        parameterID: UInt32,
        time: CMTime,
        default defaultValue: Double
    ) -> Double {
        var value: Double = defaultValue
        retrieval.getFloatValue(&value, fromParameter: parameterID, at: time)
        return value
    }

    private func intValue(
        retrieval: any FxParameterRetrievalAPI_v6,
        parameterID: UInt32,
        time: CMTime,
        default defaultValue: Int
    ) -> Int {
        var value: Int32 = Int32(defaultValue)
        retrieval.getIntValue(&value, fromParameter: parameterID, at: time)
        return Int(value)
    }

    private func boolValue(
        retrieval: any FxParameterRetrievalAPI_v6,
        parameterID: UInt32,
        time: CMTime,
        default defaultValue: Bool
    ) -> Bool {
        var value: ObjCBool = ObjCBool(defaultValue)
        retrieval.getBoolValue(&value, fromParameter: parameterID, at: time)
        return value.boolValue
    }

    /// Reads a 2D point parameter into an `(x, y)` tuple. FxPlug stores
    /// point parameters in object-normalised (0…1) coordinates with the
    /// y-axis pointing UP from the bottom-left.
    private func pointValue(
        retrieval: any FxParameterRetrievalAPI_v6,
        parameterID: UInt32,
        time: CMTime,
        defaultX: Double,
        defaultY: Double
    ) -> (Double, Double) {
        var x = defaultX
        var y = defaultY
        retrieval.getXValue(&x, yValue: &y, fromParameter: parameterID, at: time)
        return (x, y)
    }

    /// Decodes the Subject Points payload stored inside the published header
    /// custom parameter. Returns an empty set when it has never been written
    /// (the typical case until the user clicks the canvas with the OSC active).
    func loadHintPointSet(
        using retrieval: any FxParameterRetrievalAPI_v6
    ) -> HintPointSet {
        let dictionary = HeaderCustomParameterStorage.dictionaryValue(
            for: HeaderStorageKey.subjectPoints,
            using: retrieval
        )
        return HintPointSet.fromParameterDictionary(dictionary)
    }
}
