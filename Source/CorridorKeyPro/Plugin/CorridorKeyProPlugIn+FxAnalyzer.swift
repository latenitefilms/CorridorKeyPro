//
//  CorridorKeyProPlugIn+FxAnalyzer.swift
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

// Mirrors NSLock lifetime with the plug-in instance. Accessed from the main
// render thread and from background analysis callbacks, so every touch of the
// mutable state below must take this lock.
private final class AnalysisSessionState: @unchecked Sendable {
    var lock = NSLock()
    var frameDuration: CMTime = .invalid
    var firstFrameTime: CMTime = .invalid
    var frameCount: Int = 0
    var analyzedCount: Int = 0
    var screenColorRaw: Int = 0
    var inferenceResolution: Int = 0
    var matteWidth: Int = 0
    var matteHeight: Int = 0
    var mattes: [Int: Data] = [:]

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
        inferenceResolution = 0
        matteWidth = 0
        matteHeight = 0
        mattes.removeAll(keepingCapacity: false)
    }
}

// Sidecar registry that maps plug-in instances to their analysis state. Wrapped
// in a `@unchecked Sendable` class so Swift 6's data-race checker trusts the
// internal NSLock to serialise concurrent accesses from render and analysis.
private final class AnalysisStateRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [ObjectIdentifier: AnalysisSessionState] = [:]

    func state(for plugin: CorridorKeyProPlugIn) -> AnalysisSessionState {
        let identifier = ObjectIdentifier(plugin)
        lock.lock()
        defer { lock.unlock() }
        if let existing = states[identifier] {
            return existing
        }
        let fresh = AnalysisSessionState()
        states[identifier] = fresh
        return fresh
    }
}

private let analysisStateRegistry = AnalysisStateRegistry()

private func analysisState(for plugin: CorridorKeyProPlugIn) -> AnalysisSessionState {
    analysisStateRegistry.state(for: plugin)
}

extension CorridorKeyProPlugIn {

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
        let session = analysisState(for: self)
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

        let frameCount = Self.frameCount(in: analysisRange, frameDuration: frameDuration)
        let session = analysisState(for: self)
        session.lock.lock()
        session.resetLocked()
        session.frameDuration = frameDuration
        session.firstFrameTime = analysisRange.start
        session.frameCount = frameCount
        session.screenColorRaw = screenColor.rawValue
        session.inferenceResolution = inferenceResolution
        session.matteWidth = inferenceResolution
        session.matteHeight = inferenceResolution
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
        let session = analysisState(for: self)

        session.lock.lock()
        let frameDuration = session.frameDuration
        let firstFrameTime = session.firstFrameTime
        let frameCount = session.frameCount
        let screenColorRaw = session.screenColorRaw
        let storedInferenceResolution = session.inferenceResolution
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

        let extracted = try renderPipeline.extractAlphaMatteForAnalysis(
            sourceTexture: sourceTexture,
            state: analyseState,
            renderTime: frameTime,
            device: device,
            entry: entry,
            commandQueue: commandQueue
        )

        let encoded = try MatteCodec.encode(
            alpha: extracted.alpha,
            width: extracted.width,
            height: extracted.height
        )

        session.lock.lock()
        session.inferenceResolution = extracted.inferenceResolution
        session.matteWidth = extracted.width
        session.matteHeight = extracted.height
        session.mattes[frameIndex] = encoded
        session.analyzedCount = session.mattes.count
        let shouldFlush = (session.analyzedCount % 10) == 0 || session.analyzedCount == session.frameCount
        let snapshot = shouldFlush ? session.snapshotLocked() : nil
        session.lock.unlock()

        if let snapshot {
            persist(snapshot: snapshot)
        }
    }

    @objc(cleanupAnalysis:)
    func cleanupAnalysis() throws {
        let session = analysisState(for: self)
        session.lock.lock()
        let snapshot = session.snapshotLocked()
        session.lock.unlock()
        persist(snapshot: snapshot)
        PluginLog.notice(
            "Analyse: complete — \(snapshot.analyzedCount) of \(snapshot.frameCount) frame(s) cached."
        )
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
