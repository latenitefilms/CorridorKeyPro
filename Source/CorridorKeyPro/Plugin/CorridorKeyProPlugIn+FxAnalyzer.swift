//
//  CorridorKeyProPlugIn+FxAnalyzer.swift
//  Corridor Key Pro
//
//  Implements the `FxAnalyzer` protocol so the plug-in can pre-compute
//  per-frame state across the whole clip before rendering. Triggered by the
//  "Process Clip" button in the inspector, which calls `startForwardAnalysis`
//  via `FxAnalysisAPI_v2`. Final Cut Pro then drives `analyzeFrame:` for
//  every frame in the clip, letting us warm the MLX engine and cache any
//  per-frame data we need for fast playback afterwards.
//

import Foundation
import CoreMedia
import simd

extension CorridorKeyProPlugIn {

    // MARK: - Analysis time range

    @objc(desiredAnalysisTimeRange:forInputWithTimeRange:error:)
    func desiredAnalysisTimeRange(
        _ desiredRange: UnsafeMutablePointer<CMTimeRange>,
        forInputWithTimeRange inputTimeRange: CMTimeRange
    ) throws {
        desiredRange.pointee = inputTimeRange
        PluginLog.debug("Analyser requested full input time range.")
    }

    @objc(setupAnalysisForTimeRange:frameDuration:error:)
    func setupAnalysis(
        for analysisRange: CMTimeRange,
        frameDuration: CMTime
    ) throws {
        analysisLock.lock()
        analyzedFrames.removeAll(keepingCapacity: true)
        analysisFrameDuration = frameDuration
        analysisLock.unlock()
        PluginLog.notice(
            "Analyser set up for \(CMTimeGetSeconds(analysisRange.duration))s at \(CMTimeGetSeconds(frameDuration))s/frame."
        )
    }

    // MARK: - Per-frame analysis

    @objc(analyzeFrame:atTime:error:)
    func analyzeFrame(
        _ frame: FxImageTile,
        atTime frameTime: CMTime
    ) throws {
        // For now, analysis simply records the canonical reference colour per
        // frame. A follow-up will run the full render pipeline here so the
        // neural matte is cached per frame and playback becomes fast.
        let reference = SIMD3<Float>(0.08, 0.84, 0.08)
        analysisLock.lock()
        analyzedFrames.append(
            AnalyzedFrame(
                frameTime: frameTime,
                screenReference: reference,
                estimatedDifficulty: 0
            )
        )
        analysisLock.unlock()
    }

    @objc(cleanupAnalysis:)
    func cleanupAnalysis() throws {
        analysisLock.lock()
        let snapshot = analyzedFrames
        analysisLock.unlock()
        PluginLog.notice("Analyser completed with \(snapshot.count) frames.")
    }
}
