//
//  CorridorKeyToolboxPlugIn+FxAnalyzer.swift
//  Corridor Key Toolbox
//
//  Implements the `FxAnalyzer` protocol so the plug-in can run MLX inference
//  for every frame of the source clip before the user expects real-time
//  playback. Results are persisted through the hidden "Analysis Data" custom
//  parameter, which lives inside the Final Cut Pro Library — moving a project
//  between machines carries the analysis along with it.
//

import Foundation
import CoreMedia
import Metal
import simd

/// Per-plug-in analysis state. Accessed from the render thread and from the
/// background analysis callbacks, so every touch of the mutable fields below
/// must take `lock`.
///
/// Held as a stored property on `CorridorKeyToolboxPlugIn` so its lifetime is
/// pinned to the plug-in instance. Holding the state in a process-level
/// registry (as an earlier version did) leaked one full clip's worth of
/// compressed mattes per plug-in instance because the registry never learned
/// when an instance was released.
final class AnalysisSessionState: @unchecked Sendable {
    let lock = NSLock()
    var frameDuration: CMTime = .invalid
    var firstFrameTime: CMTime = .invalid
    var frameCount: Int = 0
    var analyzedCount: Int = 0
    var screenColorRaw: Int = 0
    var qualityModeRaw: Int = 0
    var inferenceResolution: Int = 0
    var matteWidth: Int = 0
    var matteHeight: Int = 0
    var mattes: [Int: Data] = [:]

    /// Previous-frame buffers used by the Phase 1 temporal-stability blend.
    /// These are the last frame's *blended* alpha (so the EMA accumulates
    /// across the clip) and the raw RGBA source at the same resolution so
    /// the motion gate can see whether a pixel actually moved.
    ///
    /// Reset when the session resets or when the inference resolution
    /// changes (which drops previous-frame sizing and forces the blender
    /// to restart from the next "current" frame).
    var previousTemporalAlpha: [Float]?
    var previousTemporalSource: [Float]?
    var previousTemporalFrameIndex: Int?

    /// Cached temporal-blend configuration captured at setup time. Reading
    /// parameter values per frame would require a retrieval API round-trip
    /// on the analysis thread, so we snapshot at the start of the pass.
    var temporalStabilityEnabled: Bool = true
    var temporalStabilityStrength: Double = 0.5

    /// Snapshot for lock-free packing into an `AnalysisData` value. Callers are
    /// expected to already hold the lock; the copy isolates the dictionary so
    /// downstream serialisation can happen outside the critical section.
    func snapshotLocked() -> AnalysisData {
        AnalysisData(
            schemaVersion: AnalysisData.currentSchemaVersion,
            frameDuration: frameDuration,
            firstFrameTime: firstFrameTime,
            frameCount: frameCount,
            analyzedCount: analyzedCount,
            screenColorRaw: screenColorRaw,
            qualityModeRaw: qualityModeRaw,
            inferenceResolution: inferenceResolution,
            matteWidth: matteWidth,
            matteHeight: matteHeight,
            mattes: mattes
        )
    }

    func resetLocked() {
        frameDuration = .invalid
        firstFrameTime = .invalid
        frameCount = 0
        analyzedCount = 0
        screenColorRaw = 0
        qualityModeRaw = 0
        inferenceResolution = 0
        matteWidth = 0
        matteHeight = 0
        mattes.removeAll(keepingCapacity: false)
        previousTemporalAlpha = nil
        previousTemporalSource = nil
        previousTemporalFrameIndex = nil
        temporalStabilityEnabled = true
        temporalStabilityStrength = 0.5
    }
}

extension CorridorKeyToolboxPlugIn {

    // MARK: - Inspector wiring (called by the +Parameters extension)

    func startForwardAnalysisPass() {
        guard let actionAPI = apiManager.api(for: (any FxCustomParameterActionAPI_v4).self) as? any FxCustomParameterActionAPI_v4 else {
            PluginLog.error("Analyse Clip: FxCustomParameterActionAPI_v4 is unavailable.")
            return
        }
        // FxPlug only vends the analysis / parameter APIs inside an action
        // bracket when the call comes from a custom UI view. Without this
        // wrapping, `apiManager.api(for:)` returns nil for both the analysis
        // and parameter-setting APIs, which is exactly what "Analyse Clip is
        // unavailable" in the logs was catching.
        actionAPI.startAction(self)
        defer { actionAPI.endAction(self) }

        guard let analysisAPI = apiManager.api(for: (any FxAnalysisAPI_v2).self) as? any FxAnalysisAPI_v2 else {
            PluginLog.error("Analyse Clip: FxAnalysisAPI_v2 is unavailable.")
            return
        }
        let state = analysisAPI.analysisStateForEffect()
        if state == kFxAnalysisState_AnalysisRequested || state == kFxAnalysisState_AnalysisStarted {
            PluginLog.notice("Analyse Clip: analysis already in progress; ignoring duplicate click.")
            return
        }
        do {
            try analysisAPI.startForwardAnalysis(kFxAnalysisLocation_GPU)
            PluginLog.notice("Analyse Clip: forward analysis requested.")
        } catch {
            PluginLog.error("Analyse Clip: failed to start forward analysis: \(error.localizedDescription)")
        }
    }

    func clearAnalysisCache() {
        let session = analysisSession
        session.lock.lock()
        session.resetLocked()
        session.lock.unlock()

        guard let actionAPI = apiManager.api(for: (any FxCustomParameterActionAPI_v4).self) as? any FxCustomParameterActionAPI_v4 else {
            PluginLog.error("Reset Analysis: FxCustomParameterActionAPI_v4 is unavailable.")
            return
        }
        actionAPI.startAction(self)
        defer { actionAPI.endAction(self) }

        guard let setAPI = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            PluginLog.error("Reset Analysis: FxParameterSettingAPI_v5 is unavailable.")
            return
        }
        setAPI.setCustomParameterValue(
            NSDictionary(),
            toParameter: ParameterIdentifier.analysisData,
            at: CMTime.zero
        )
        PluginLog.notice("Analysis cache cleared.")
    }

    // MARK: - FxAnalyzer conformance

    @objc(desiredAnalysisTimeRange:forInputWithTimeRange:error:)
    func desiredAnalysisTimeRange(
        _ desiredRange: UnsafeMutablePointer<CMTimeRange>,
        forInputWithTimeRange inputTimeRange: CMTimeRange
    ) throws {
        desiredRange.pointee = inputTimeRange
        PluginLog.notice(
            "Analyse: desired time range = \(CMTimeGetSeconds(inputTimeRange.duration))s."
        )
    }

    @objc(setupAnalysisForTimeRange:frameDuration:error:)
    func setupAnalysis(
        for analysisRange: CMTimeRange,
        frameDuration: CMTime
    ) throws {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_APIUnavailable,
                userInfo: [NSLocalizedDescriptionKey: "Could not access parameter retrieval API during analysis setup."]
            )
        }

        let screenColor: ScreenColor = {
            var raw: Int32 = 0
            guard retrieval.getIntValue(&raw, fromParameter: ParameterIdentifier.screenColor, at: analysisRange.start) else {
                return .green
            }
            return ScreenColor(rawValue: Int(raw)) ?? .green
        }()
        let qualityMode: QualityMode = {
            var raw: Int32 = 0
            guard retrieval.getIntValue(&raw, fromParameter: ParameterIdentifier.qualityMode, at: analysisRange.start) else {
                return .automatic
            }
            return QualityMode(rawValue: Int(raw)) ?? .automatic
        }()

        // We don't know the clip's intrinsic resolution at setup time, so use
        // the baseline long edge to pick an inference rung. Automatic settles
        // for 1024px which is the practical sweet spot between storage cost
        // and edge fidelity; the user can still pick Ultra/Maximum manually.
        let longEdge = Int(1920)
        let inferenceResolution = qualityMode.resolvedInferenceResolution(forLongEdge: longEdge)

        // Temporal stability settings are also snapshotted at setup so the
        // per-frame analyser doesn't hit the parameter retrieval API from a
        // background thread. These read with safe defaults if the project
        // predates the parameter.
        let temporalEnabled: Bool = {
            var raw = ObjCBool(true)
            retrieval.getBoolValue(&raw, fromParameter: ParameterIdentifier.temporalStabilityEnabled, at: analysisRange.start)
            return raw.boolValue
        }()
        let temporalStrength: Double = {
            var value: Double = 0.5
            retrieval.getFloatValue(&value, fromParameter: ParameterIdentifier.temporalStabilityStrength, at: analysisRange.start)
            return value
        }()

        let frameCount = Self.frameCount(in: analysisRange, frameDuration: frameDuration)
        let session = analysisSession
        session.lock.lock()
        session.resetLocked()
        session.frameDuration = frameDuration
        session.firstFrameTime = analysisRange.start
        session.frameCount = frameCount
        session.screenColorRaw = screenColor.rawValue
        session.qualityModeRaw = qualityMode.rawValue
        session.inferenceResolution = inferenceResolution
        session.matteWidth = inferenceResolution
        session.matteHeight = inferenceResolution
        session.temporalStabilityEnabled = temporalEnabled
        session.temporalStabilityStrength = temporalStrength
        session.lock.unlock()

        PluginLog.notice(
            "Analyse setup: \(frameCount) frame(s) at \(inferenceResolution)px, screen=\(screenColor.displayName)."
        )
    }

    @objc(analyzeFrame:atTime:error:)
    func analyzeFrame(
        _ frame: FxImageTile,
        atTime frameTime: CMTime
    ) throws {
        let session = analysisSession

        session.lock.lock()
        let frameDuration = session.frameDuration
        let firstFrameTime = session.firstFrameTime
        let frameCount = session.frameCount
        let screenColorRaw = session.screenColorRaw
        let storedInferenceResolution = session.inferenceResolution
        let temporalEnabled = session.temporalStabilityEnabled
        let temporalStrength = session.temporalStabilityStrength
        let cachedPreviousAlpha = session.previousTemporalAlpha
        let cachedPreviousSource = session.previousTemporalSource
        let cachedPreviousFrameIndex = session.previousTemporalFrameIndex
        session.lock.unlock()

        guard frameCount > 0 else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_AnalysisError,
                userInfo: [NSLocalizedDescriptionKey: "Analysis was not initialised before analyzeFrame was called."]
            )
        }
        guard let frameIndex = Self.frameIndex(
            for: frameTime,
            firstFrameTime: firstFrameTime,
            frameDuration: frameDuration,
            frameCount: frameCount
        ) else {
            PluginLog.notice("Analyse: skipping out-of-range frame at \(CMTimeGetSeconds(frameTime))s.")
            return
        }

        let deviceCache = MetalDeviceCache.shared
        guard let device = deviceCache.device(forRegistryID: frame.deviceRegistryID) else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_AnalysisError,
                userInfo: [NSLocalizedDescriptionKey: "Analyse: unknown GPU with registry id \(frame.deviceRegistryID)."]
            )
        }
        let entry = try deviceCache.entry(for: device)
        guard let commandQueue = entry.borrowCommandQueue() else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_AnalysisError,
                userInfo: [NSLocalizedDescriptionKey: "Analyse: no Metal command queue is available."]
            )
        }
        defer { entry.returnCommandQueue(commandQueue) }

        guard let sourceTexture = frame.metalTexture(for: device) else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_AnalysisError,
                userInfo: [NSLocalizedDescriptionKey: "Analyse: analysis frame had no Metal texture."]
            )
        }

        let analyseState = PluginStateData(
            screenColor: ScreenColor(rawValue: screenColorRaw) ?? .green,
            qualityMode: QualityMode.allCases.first(where: { $0.resolvedInferenceResolution(forLongEdge: 1920) == storedInferenceResolution })
                ?? .automatic
        )

        let workingGamut: WorkingColorGamut
        if let gamutAPI = apiManager.api(for: (any FxColorGamutAPI_v2).self) as? any FxColorGamutAPI_v2 {
            workingGamut = ColorGamutMatrix.gamut(fromColorPrimariesRaw: UInt(gamutAPI.colorPrimaries()))
        } else {
            workingGamut = .rec709
        }

        // Request a source readback only when temporal stability is armed
        // and we either already have a previous frame (immediate blend) or
        // might need one on the next frame. The readback adds a
        // `.shared`-storage blit + 64 MB (Maximum rung) hand-off; skipping
        // it when the user has disabled the feature removes that cost.
        let needsTemporalReadback = temporalEnabled && temporalStrength > 0
        let extracted = try renderPipeline.extractAlphaMatteForAnalysis(
            sourceTexture: sourceTexture,
            state: analyseState,
            workingGamut: workingGamut,
            renderTime: frameTime,
            device: device,
            entry: entry,
            commandQueue: commandQueue,
            readbackSource: needsTemporalReadback
        )

        // Apply the temporal blend when we have a valid previous frame at
        // the same resolution. A change of inference resolution (user flipped
        // Quality mid-pass) invalidates the prior state — drop it silently
        // and let the next frame seed the EMA afresh.
        var currentAlpha = extracted.alpha
        if needsTemporalReadback,
           let previousAlpha = cachedPreviousAlpha,
           let previousSource = cachedPreviousSource,
           let previousFrameIndex = cachedPreviousFrameIndex,
           let currentSource = extracted.source,
           previousAlpha.count == currentAlpha.count,
           previousSource.count == currentSource.count,
           previousFrameIndex + 1 == frameIndex {
            TemporalBlender.applyInPlace(
                currentAlpha: &currentAlpha,
                previousAlpha: previousAlpha,
                currentSource: currentSource,
                previousSource: previousSource,
                width: extracted.width,
                height: extracted.height,
                configuration: TemporalBlender.Configuration(
                    strength: Float(temporalStrength),
                    motionThreshold: 0.05
                )
            )
        }

        let encoded = try MatteCodec.encode(
            alpha: currentAlpha,
            width: extracted.width,
            height: extracted.height
        )

        session.lock.lock()
        session.inferenceResolution = extracted.inferenceResolution
        session.matteWidth = extracted.width
        session.matteHeight = extracted.height
        session.mattes[frameIndex] = encoded
        session.analyzedCount = session.mattes.count
        if needsTemporalReadback {
            session.previousTemporalAlpha = currentAlpha
            session.previousTemporalSource = extracted.source
            session.previousTemporalFrameIndex = frameIndex
        } else {
            session.previousTemporalAlpha = nil
            session.previousTemporalSource = nil
            session.previousTemporalFrameIndex = nil
        }
        let shouldFlush = (session.analyzedCount % 10) == 0 || session.analyzedCount == session.frameCount
        let snapshot = shouldFlush ? session.snapshotLocked() : nil
        session.lock.unlock()

        if let snapshot {
            persist(snapshot: snapshot)
        }
    }

    @objc(cleanupAnalysis:)
    func cleanupAnalysis() throws {
        let session = analysisSession
        session.lock.lock()
        let snapshot = session.snapshotLocked()
        session.lock.unlock()
        persist(snapshot: snapshot)
        // The persisted copy inside the FCP Library is now the source of
        // truth for the render path — drop the in-memory mattes to free up
        // the working set. For a 2048px matte cache on a long clip this can
        // reclaim hundreds of MB. Temporal state also drops here; the EMA
        // belongs to a single analysis pass, not the lifetime of the plug-in.
        session.lock.lock()
        session.mattes.removeAll(keepingCapacity: false)
        session.previousTemporalAlpha = nil
        session.previousTemporalSource = nil
        session.previousTemporalFrameIndex = nil
        session.lock.unlock()
        PluginLog.notice(
            "Analyse: complete — \(snapshot.analyzedCount) of \(snapshot.frameCount) frame(s) cached."
        )
    }

    // MARK: - Live progress for the inspector header

    /// Lightweight view into the in-memory session state used by the
    /// inspector bridge while analysis is running. Reading the persisted
    /// `AnalysisData` would only catch up every 10 frames (and wouldn't
    /// surface the chosen inference resolution until the first flush), which
    /// left the header stuck at "Analysing at 0px…" for long clips.
    func liveAnalysisProgress() -> (analysed: Int, total: Int, resolution: Int)? {
        let session = analysisSession
        session.lock.lock()
        defer { session.lock.unlock() }
        guard session.frameCount > 0 else { return nil }
        return (session.analyzedCount, session.frameCount, session.inferenceResolution)
    }

    // MARK: - Persistence + lookup helpers

    /// Writes the current in-memory cache back to the hidden custom
    /// parameter so Final Cut Pro picks it up on the next render request.
    private func persist(snapshot: AnalysisData) {
        guard let setAPI = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            PluginLog.error("Persist analysis: FxParameterSettingAPI_v5 is unavailable.")
            return
        }
        setAPI.setCustomParameterValue(
            snapshot.asParameterDictionary(),
            toParameter: ParameterIdentifier.analysisData,
            at: CMTime.zero
        )
    }

    /// Loads the persisted cache for a render call. Used by `pluginState` to
    /// surface the current frame's compressed matte into the render blob.
    func loadAnalysisData(using retrieval: any FxParameterRetrievalAPI_v6) -> AnalysisData? {
        var rawValue: (any NSCopying & NSObjectProtocol & NSSecureCoding)?
        retrieval.getCustomParameterValue(
            &rawValue,
            fromParameter: ParameterIdentifier.analysisData,
            at: CMTime.zero
        )
        return AnalysisData.fromParameterDictionary(rawValue as? NSDictionary)
    }

    // MARK: - Frame index math

    static func frameCount(in range: CMTimeRange, frameDuration: CMTime) -> Int {
        let durationSeconds = CMTimeGetSeconds(frameDuration)
        guard durationSeconds > 0 else { return 0 }
        let totalSeconds = CMTimeGetSeconds(range.duration)
        let count = Int((totalSeconds / durationSeconds).rounded(.up))
        return max(0, count)
    }

    static func frameIndex(
        for renderTime: CMTime,
        firstFrameTime: CMTime,
        frameDuration: CMTime,
        frameCount: Int
    ) -> Int? {
        let durationSeconds = CMTimeGetSeconds(frameDuration)
        guard durationSeconds > 0 else { return nil }
        let delta = CMTimeGetSeconds(CMTimeSubtract(renderTime, firstFrameTime))
        let index = Int((delta / durationSeconds).rounded())
        guard index >= 0, index < frameCount else { return nil }
        return index
    }
}
