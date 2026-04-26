//
//  AnalysisRunner.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Mirrors `CorridorKeyToolboxPlugIn+FxAnalyzer` for the standalone
//  editor: walks every frame of a clip, hands the source to MLX via
//  `StandaloneRenderEngine.extractMatteBlob`, and stores the resulting
//  matte blobs in an in-memory cache keyed by frame index. The render
//  preview then short-circuits MLX by feeding cached mattes into
//  `RenderPipeline.renderToTexture` exactly the way the FxPlug renders
//  cached frames inside Final Cut Pro.
//
//  Runs as a Swift `Task` driven by `EditorViewModel`. Yields
//  cooperative cancellation between frames so the user can stop the
//  pass at any time. Reports progress via an `AsyncStream` so the
//  inspector progress UI stays current without polling.
//

import Foundation
import CoreMedia
import CoreVideo

/// Per-frame storage built up during the analysis pass. Exposed via
/// `EditorProject.matteCache` so previews can reach in for a cached
/// matte by frame index instead of triggering MLX on every scrub.
struct MatteCacheEntry: @unchecked Sendable {
    let blob: Data
    let width: Int
    let height: Int
    let inferenceResolution: Int
}

/// Status events the runner pushes during a pass. Drives the
/// progress bar and the success / failure alerts in the editor.
enum AnalysisRunnerEvent: @unchecked Sendable {
    /// Total frame count became known (or changed).
    case totalFramesResolved(Int)
    /// One frame was processed. `frameIndex` is zero-based, increasing
    /// by one each callback. `engineDescription` reflects whether MLX
    /// or the rough-matte fallback handled the frame.
    case frameProcessed(frameIndex: Int, engineDescription: String, entry: MatteCacheEntry)
    /// Pass finished without error. Includes the total elapsed seconds.
    case completed(elapsedSeconds: Double)
    /// Pass aborted with an error.
    case failed(String)
    /// User cancelled the pass.
    case cancelled
}

actor AnalysisRunner {

    private let renderEngine: StandaloneRenderEngine
    private let videoSource: VideoSource

    init(renderEngine: StandaloneRenderEngine, videoSource: VideoSource) {
        self.renderEngine = renderEngine
        self.videoSource = videoSource
    }

    /// Runs the analysis pass to completion or until the wrapping task
    /// is cancelled. Each frame's matte blob is delivered through
    /// `eventHandler` so the editor can store it directly in its
    /// `EditorProject` cache without the runner needing to know about
    /// the project type.
    func run(
        state: PluginStateData,
        eventHandler: @Sendable (AnalysisRunnerEvent) -> Void
    ) async {
        let startedAt = Date()
        let info = videoSource.info
        let frameCount = videoSource.totalFrameCount()
        let frameRate = max(Double(info.nominalFrameRate), 0.001)
        let timescale = info.timescale
        eventHandler(.totalFramesResolved(frameCount))

        // Build the AVAssetReader once for the whole pass — far faster
        // than reseeking per frame via AVAssetImageGenerator.
        let frameReader: VideoFrameReader
        do {
            frameReader = try videoSource.makeFrameReader(startTime: .zero)
        } catch {
            eventHandler(.failed(String(describing: error)))
            return
        }
        defer { frameReader.cancel() }

        var frameIndex = 0
        while !Task.isCancelled {
            let frame: ReaderFrame?
            do {
                frame = try frameReader.nextFrame()
            } catch {
                eventHandler(.failed(String(describing: error)))
                return
            }
            guard let frame else { break }

            let renderTime: CMTime
            if frame.presentationTime.isValid && frame.presentationTime.isNumeric {
                renderTime = frame.presentationTime
            } else {
                renderTime = CMTime(
                    seconds: Double(frameIndex) / frameRate,
                    preferredTimescale: timescale
                )
            }

            let output: AnalysisFrameOutput
            do {
                output = try renderEngine.extractMatteBlob(
                    source: frame.pixelBuffer,
                    state: state,
                    renderTime: renderTime
                )
            } catch {
                eventHandler(.failed(String(describing: error)))
                return
            }

            let entry = MatteCacheEntry(
                blob: output.blob,
                width: output.width,
                height: output.height,
                inferenceResolution: output.inferenceResolution
            )
            eventHandler(.frameProcessed(
                frameIndex: frameIndex,
                engineDescription: output.engineDescription,
                entry: entry
            ))
            frameIndex += 1
            // Yield cooperative cancellation between frames; otherwise
            // a tight inference loop blocks Task.checkCancellation.
            await Task.yield()
        }

        if Task.isCancelled {
            eventHandler(.cancelled)
            return
        }
        let elapsed = Date().timeIntervalSince(startedAt)
        eventHandler(.completed(elapsedSeconds: elapsed))
    }
}
