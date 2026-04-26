//
//  EditorViewModel.swift
//  Corridor Key Toolbox — Standalone Editor
//
//  Single source of truth for the editor window. Holds:
//
//  * The currently-loaded `VideoSource` (or nil if no clip has been
//    imported).
//  * The live `PluginStateData` parameters bound to the inspector.
//  * The matte cache populated by the analysis pass.
//  * The current preview frame and any in-flight render task.
//  * Transport state (playhead, playing / paused).
//  * Analysis and export progress.
//
//  All UI bindings flow through `@Observable` properties so SwiftUI
//  invalidates only the panes whose data actually changed.
//

import Foundation
import Observation
import Metal
import CoreMedia
import AVFoundation
import Combine
import SwiftUI

/// Coarse states the editor can be in. Drives empty-state vs editor
/// chrome and which actions are enabled.
enum EditorPhase: @unchecked Sendable, Equatable {
    case noClipLoaded
    case loadingClip(URL)
    case ready
    case loadFailed(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

/// Status of the analyse-clip pass. Drives the inspector progress UI.
enum EditorAnalysisStatus: @unchecked Sendable, Equatable {
    case idle
    case running(processed: Int, total: Int)
    case completed(elapsedSeconds: Double, totalFrames: Int)
    case failed(String)
    case cancelled

    var inProgress: Bool {
        if case .running = self { return true }
        return false
    }
}

/// Status of the export pass. Drives the export-sheet progress UI.
enum EditorExportStatus: @unchecked Sendable, Equatable {
    case idle
    case running(processed: Int, total: Int)
    case completed(URL)
    case failed(String)
    case cancelled

    var inProgress: Bool {
        if case .running = self { return true }
        return false
    }
}

@MainActor
@Observable
final class EditorViewModel {

    // MARK: - Foundational state

    @ObservationIgnored
    let renderEngine: StandaloneRenderEngine

    var phase: EditorPhase = .noClipLoaded
    /// All user-editable parameters. Bound directly to the inspector
    /// controls; mutating any property triggers a preview re-render via
    /// `parameterDidChange`.
    var state: PluginStateData = PluginStateData()
    /// Loaded clip metadata. `nil` while `.noClipLoaded`.
    var clipInfo: VideoSourceInfo?
    /// Most recently rendered frame, ready to draw on the preview
    /// surface. Updated whenever the playhead moves or a parameter
    /// changes.
    var latestPreview: PreviewFrame?
    /// Current playhead position in source time. Snapped to source
    /// frame boundaries on every set.
    var playheadTime: CMTime = .zero
    /// Total frame count for the loaded clip. Drives the scrubber
    /// granularity and analyse / export progress totals.
    var totalFrames: Int = 1
    /// Status of the most recent analyse pass.
    var analysisStatus: EditorAnalysisStatus = .idle
    /// Status of the most recent export pass.
    var exportStatus: EditorExportStatus = .idle
    /// In-memory matte cache populated by `runAnalysis`. Keyed by
    /// frame index.
    var matteCache: [Int: MatteCacheEntry] = [:]
    /// MLX warm-up status for the inspector badge.
    var warmupStatus: WarmupStatus = .cold
    /// User-readable description of the last render's backend (e.g.
    /// "Source Pass-Through", "Analysed Cache (1024px)"). Surfaced
    /// in the status row beneath the preview.
    var renderBackendDescription: String = "—"
    /// Effective render size, used to size the preview surface and
    /// label the export panel.
    var renderSize: CGSize = .zero
    /// Last-known engine description (mirrors the FxPlug
    /// "Inference Backend" badge — surfaces "MLX" vs "Rough matte").
    var lastEngineDescription: String = "—"

    // MARK: - Internal state

    @ObservationIgnored
    private var videoSource: VideoSource?
    @ObservationIgnored
    private var inflightRenderTask: Task<Void, Never>?
    @ObservationIgnored
    private var analysisTask: Task<Void, Never>?
    @ObservationIgnored
    private var exportTask: Task<Void, Never>?
    /// Debounces preview re-renders during slider drags so the GPU
    /// isn't flooded with intermediate states. Reset every parameter
    /// edit; whoever fires last wins.
    @ObservationIgnored
    private var pendingPreviewTask: Task<Void, Never>?

    init(renderEngine: StandaloneRenderEngine) {
        self.renderEngine = renderEngine
    }

    // MARK: - Clip loading

    /// Loads a clip into the editor. Existing matte cache is dropped.
    /// On success transitions to `.ready` and renders the first frame
    /// for preview.
    func loadClip(at url: URL) async {
        cancelInflightWork()
        phase = .loadingClip(url)
        do {
            let source = try await VideoSource(url: url)
            self.videoSource = source
            let info = source.info
            self.clipInfo = info
            self.totalFrames = source.totalFrameCount()
            self.renderSize = info.renderSize
            self.matteCache.removeAll(keepingCapacity: true)
            self.analysisStatus = .idle
            self.exportStatus = .idle
            self.playheadTime = .zero
            self.phase = .ready
            beginWarmupForCurrentQuality()
            renderPreview(at: .zero)
        } catch {
            self.phase = .loadFailed(error.localizedDescription)
        }
    }

    /// Closes the current clip and returns the editor to the empty
    /// state. Cancels any in-flight analyse / export work.
    func closeClip() {
        cancelInflightWork()
        videoSource = nil
        clipInfo = nil
        latestPreview = nil
        matteCache.removeAll()
        analysisStatus = .idle
        exportStatus = .idle
        playheadTime = .zero
        renderBackendDescription = "—"
        lastEngineDescription = "—"
        phase = .noClipLoaded
    }

    // MARK: - Transport

    /// Moves the playhead to a normalised position (0…1). Snaps to the
    /// nearest source frame and renders a preview.
    func scrub(toNormalized fraction: Double) {
        guard let source = videoSource, let clipInfo else { return }
        let clamped = min(max(fraction, 0), 1)
        let durationSeconds = clipInfo.duration.seconds
        let target = CMTime(
            seconds: clamped * durationSeconds,
            preferredTimescale: clipInfo.timescale
        )
        Task { [source] in
            let snapped = source.nearestFrameTime(to: target)
            await MainActor.run {
                self.playheadTime = snapped
                self.renderPreview(at: snapped)
            }
        }
    }

    /// Moves the playhead by an integer number of source frames.
    /// Wraps inputs to stay within the clip duration. Used by the
    /// transport's step buttons.
    func step(byFrames delta: Int) {
        guard let clipInfo else { return }
        let frameRate = max(Double(clipInfo.nominalFrameRate), 0.001)
        let currentFrame = Int((playheadTime.seconds * frameRate).rounded())
        let targetFrame = max(0, min(totalFrames - 1, currentFrame + delta))
        let target = CMTime(
            seconds: Double(targetFrame) / frameRate,
            preferredTimescale: clipInfo.timescale
        )
        playheadTime = target
        renderPreview(at: target)
    }

    // MARK: - Preview rendering

    /// Schedules a preview render for the given source time. Cancels
    /// any pending render so the latest parameter edit always wins.
    func renderPreview(at time: CMTime) {
        guard let source = videoSource else { return }
        let frameTime = time
        let frameIndex = currentFrameIndex(at: frameTime)
        var stateForFrame = state
        if let cached = matteCache[frameIndex] {
            stateForFrame.cachedMatteBlob = cached.blob
            stateForFrame.cachedMatteInferenceResolution = cached.inferenceResolution
        }
        stateForFrame.destinationLongEdgePixels = max(
            Int(renderSize.width.rounded()),
            Int(renderSize.height.rounded())
        )
        let renderEngine = renderEngine

        pendingPreviewTask?.cancel()
        pendingPreviewTask = Task { [weak self] in
            guard let self else { return }
            let pixelBuffer: CVPixelBuffer
            do {
                pixelBuffer = try await source.makeFrame(atTime: frameTime)
            } catch {
                if Task.isCancelled { return }
                self.phase = .loadFailed(error.localizedDescription)
                return
            }
            if Task.isCancelled { return }
            do {
                let result = try renderEngine.render(
                    source: pixelBuffer,
                    state: stateForFrame,
                    renderTime: frameTime
                )
                if Task.isCancelled { return }
                self.latestPreview = PreviewFrame(
                    texture: result.destinationTexture,
                    pixelBuffer: result.destinationPixelBuffer,
                    presentationTime: frameTime
                )
                self.renderBackendDescription = result.report.backendDescription
                self.lastEngineDescription = result.report.guideSourceDescription
            } catch {
                if Task.isCancelled { return }
                self.phase = .loadFailed(error.localizedDescription)
            }
        }
    }

    /// Re-renders the current preview frame. Called whenever the user
    /// edits a parameter so the change is reflected immediately.
    func parameterDidChange() {
        renderPreview(at: playheadTime)
    }

    // MARK: - Analysis

    /// Starts the analyse-clip pass. The matte cache is rebuilt from
    /// scratch — the editor never blends partial caches into a result.
    func runAnalysis() {
        guard let source = videoSource else { return }
        cancelAnalysis()
        let runner = AnalysisRunner(renderEngine: renderEngine, videoSource: source)
        let stateSnapshot = state
        let startingPlayhead = playheadTime
        analysisStatus = .running(processed: 0, total: totalFrames)
        matteCache.removeAll(keepingCapacity: true)
        beginWarmupForCurrentQuality()

        // Bridge the runner's `@Sendable` event callback to a main-actor
        // event loop via an AsyncStream. This keeps the event sink off of
        // self entirely — only the surrounding for-await loop touches the
        // view model, and that loop is already main-actor-isolated.
        let (eventStream, continuation) = AsyncStream<AnalysisRunnerEvent>.makeStream()
        analysisTask = Task { [weak self] in
            let runTask = Task.detached(priority: .userInitiated) {
                await runner.run(state: stateSnapshot) { event in
                    continuation.yield(event)
                }
                continuation.finish()
            }
            for await event in eventStream {
                guard let self else { break }
                self.handleAnalysisEvent(event, fallbackTotalFrames: self.totalFrames)
            }
            _ = await runTask.value
            self?.analysisTask = nil
            // After analysis completes, refresh the preview so the
            // current playhead picks up its newly-cached matte.
            self?.renderPreview(at: startingPlayhead)
        }
    }

    /// Cancels an in-flight analyse pass. Already-cached frames stay
    /// in the cache so the user can still preview them; the rest of
    /// the clip will re-trigger pass-through.
    func cancelAnalysis() {
        analysisTask?.cancel()
        analysisTask = nil
        if case .running = analysisStatus {
            analysisStatus = .cancelled
        }
    }

    @MainActor
    private func handleAnalysisEvent(
        _ event: AnalysisRunnerEvent,
        fallbackTotalFrames: Int
    ) {
        switch event {
        case .totalFramesResolved(let total):
            self.totalFrames = max(total, 1)
            self.analysisStatus = .running(processed: 0, total: max(total, 1))
        case .frameProcessed(let frameIndex, let engineDescription, let entry):
            self.matteCache[frameIndex] = entry
            self.lastEngineDescription = engineDescription
            let total: Int
            if case .running(_, let resolvedTotal) = analysisStatus {
                total = resolvedTotal
            } else {
                total = fallbackTotalFrames
            }
            self.analysisStatus = .running(processed: frameIndex + 1, total: total)
        case .completed(let elapsed):
            self.analysisStatus = .completed(
                elapsedSeconds: elapsed,
                totalFrames: max(self.matteCache.count, 1)
            )
        case .failed(let message):
            self.analysisStatus = .failed(message)
        case .cancelled:
            self.analysisStatus = .cancelled
        }
    }

    // MARK: - Export

    /// Starts a ProRes 4444 export. The destination is the URL the
    /// user picked in the save panel; this method assumes that URL is
    /// already accessible to the wrapper app's sandbox.
    func runExport(options: ExportOptions) {
        guard let source = videoSource else { return }
        cancelExport()
        let snapshot = ExportProjectSnapshot(state: state, cachedMattes: matteCache)
        let exporter = ProResExporter(
            renderEngine: renderEngine,
            videoSource: source,
            project: snapshot
        )
        let totalFramesSnapshot = totalFrames
        exportStatus = .running(processed: 0, total: totalFramesSnapshot)

        let (eventStream, continuation) = AsyncStream<ExportRunnerEvent>.makeStream()
        exportTask = Task { [weak self] in
            let runTask = Task.detached(priority: .userInitiated) {
                await exporter.run(options: options) { event in
                    continuation.yield(event)
                }
                continuation.finish()
            }
            for await event in eventStream {
                guard let self else { break }
                self.handleExportEvent(event, fallbackTotalFrames: totalFramesSnapshot)
            }
            _ = await runTask.value
            self?.exportTask = nil
        }
    }

    /// Cancels an in-flight export pass.
    func cancelExport() {
        exportTask?.cancel()
        exportTask = nil
        if case .running = exportStatus {
            exportStatus = .cancelled
        }
    }

    @MainActor
    private func handleExportEvent(
        _ event: ExportRunnerEvent,
        fallbackTotalFrames: Int
    ) {
        switch event {
        case .started(let total):
            self.exportStatus = .running(processed: 0, total: max(total, 1))
        case .frameWritten(let frameIndex):
            let total: Int
            if case .running(_, let resolvedTotal) = exportStatus {
                total = resolvedTotal
            } else {
                total = fallbackTotalFrames
            }
            self.exportStatus = .running(processed: frameIndex + 1, total: total)
        case .completed(let url):
            self.exportStatus = .completed(url)
        case .failed(let message):
            self.exportStatus = .failed(message)
        case .cancelled:
            self.exportStatus = .cancelled
        }
    }

    // MARK: - Helpers

    private func beginWarmupForCurrentQuality() {
        let longEdge = max(Int(renderSize.width.rounded()), Int(renderSize.height.rounded()))
        guard longEdge > 0 else { return }
        let resolution = state.qualityMode.resolvedInferenceResolution(forLongEdge: longEdge)
        do {
            try renderEngine.beginWarmup(forResolution: resolution)
            warmupStatus = renderEngine.warmupStatus(forResolution: resolution)
        } catch {
            // Warm-up is best-effort. If it fails the analyse pass
            // surfaces the same error the moment it tries to use MLX.
        }
    }

    private func cancelInflightWork() {
        pendingPreviewTask?.cancel(); pendingPreviewTask = nil
        cancelAnalysis()
        cancelExport()
    }

    private func currentFrameIndex(at time: CMTime) -> Int {
        guard let clipInfo else { return 0 }
        let fps = max(Double(clipInfo.nominalFrameRate), 0.001)
        return Int((time.seconds * fps).rounded())
    }
}

/// Light value type holding the bytes the preview surface currently
/// shows. Holding the source pixel buffer keeps the IOSurface (and
/// therefore the texture's underlying memory) alive until the next
/// frame replaces this value.
struct PreviewFrame {
    let texture: any MTLTexture
    let pixelBuffer: CVPixelBuffer
    let presentationTime: CMTime
}
