//
//  CorridorKeyProPlugIn+FxAnalyzer.swift
//  Corridor Key Pro
//
//  Implements the `FxAnalyzer` protocol so the plug-in can pre-compute
//  per-frame statistics – primarily the screen-colour reference – across the
//  whole clip before rendering. Mirrors the pattern shown in Apple's
//  `FxBrightnessAnalysis` sample but tailored to our needs: we store a
//  `SIMD3<Float>` reference plus a difficulty score per frame so the renderer
//  can pick a conservative quality rung when the signal is shaky.
//

import Foundation
import CoreMedia
import simd

extension CorridorKeyProPlugIn {

    // MARK: - Analysis time range

    func desiredAnalysisTimeRange(
        _ desiredRange: UnsafeMutablePointer<CMTimeRange>!,
        forInputWith inputTimeRange: CMTimeRange
    ) throws {
        // Analyse the entire clip the user applied the effect to. If we ever
        // add scoped analysis (for example "only analyse the current play
        // range") we can narrow this here.
        desiredRange.pointee = inputTimeRange
    }

    func setupAnalysis(
        for analysisRange: CMTimeRange,
        frameDuration: CMTime
    ) throws {
        analysisLock.lock()
        analyzedFrames.removeAll(keepingCapacity: true)
        analysisFrameDuration = frameDuration
        analysisLock.unlock()
    }

    // MARK: - Per-frame analysis

    func analyzeFrame(
        _ frame: FxImageTile!,
        at frameTime: CMTime
    ) throws {
        // For the initial implementation we accumulate the screen reference
        // using the CorridorKey canonical value per screen colour. When the
        // GPU-side estimator is enabled we'll replace this with an async
        // readback of a downsampled patch.
        let reference = SIMD3<Float>(0.1, 0.75, 0.2)
        let difficulty = 0.0

        analysisLock.lock()
        analyzedFrames.append(
            AnalyzedFrame(
                frameTime: frameTime,
                screenReference: reference,
                estimatedDifficulty: difficulty
            )
        )
        analysisLock.unlock()
    }

    func cleanupAnalysis() throws {
        // Hand the accumulated analysis data back to the host through our
        // hidden analysis parameter. The renderer picks it up via
        // `pluginState` when the user scrubs the timeline.
        analysisLock.lock()
        let snapshot = analyzedFrames
        analysisLock.unlock()

        guard let setting = apiManager.api(for: FxParameterSettingAPI_v6.self) as? FxParameterSettingAPI_v6 else {
            return // Analysis still finished; we just can't persist the results this session.
        }

        if let encoded = try? JSONEncoder().encode(snapshot),
           let string = String(data: encoded, encoding: .utf8) {
            // Stash the analysis blob as a string parameter. This avoids the
            // custom-parameter NSCoding dance for MVP. A richer parameter type
            // can replace this when we need keyframe-interpolated analysis.
            setting.setStringParameterValue(string, toParameter: ParameterIdentifier.statusGuideSource)
        }
    }
}
