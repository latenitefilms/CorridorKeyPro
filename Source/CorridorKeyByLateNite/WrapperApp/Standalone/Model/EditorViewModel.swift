//
//  EditorViewModel.swift
//  CorridorKey by LateNite — Standalone Editor
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

/// Backdrop the preview surface composites the keyed image over.
/// Defaults to a transparency-aware checkerboard so users can read
/// the matte at a glance, with solid-colour options for alternate
/// reference looks. The two final cases — `.customColor` and
/// `.customImage` — pair with `EditorViewModel.customBackdropColor`
/// and `customBackdropTexture` respectively; the enum identity is
/// what the Picker selects on, while the supporting data lives on
/// the view model so the inline radio rows still get a stable tag.
enum PreviewBackdrop: Hashable, Sendable, CaseIterable, Identifiable {
    case checkerboard
    case white
    case black
    case yellow
    case red
    case customColor
    case customImage

    var id: Self { self }

    var displayName: String {
        switch self {
        case .checkerboard: return "Checkerboard"
        case .white: return "White"
        case .black: return "Black"
        case .yellow: return "Yellow"
        case .red: return "Red"
        case .customColor: return "Custom Colour…"
        case .customImage: return "Custom Image…"
        }
    }

    var systemImage: String {
        switch self {
        case .checkerboard: return "checkerboard.rectangle"
        case .white: return "rectangle.fill"
        case .black: return "rectangle.fill"
        case .yellow: return "rectangle.fill"
        case .red: return "rectangle.fill"
        case .customColor: return "paintpalette"
        case .customImage: return "photo"
        }
    }
}

/// Linear RGB triplet a custom backdrop is rendered with. Stored on
/// the view model and persisted via UserDefaults so the user's last
/// pick survives a launch. Lives as a value type rather than
/// SwiftUI's `Color` so it's `Codable` and `Sendable` without
/// pulling in environment lookups.
struct BackdropColor: Hashable, Sendable, Codable {
    var red: Double
    var green: Double
    var blue: Double

    /// Neutral mid-grey default — readable on top of most clip
    /// looks without biasing the user's eye toward any hue.
    static let `default` = BackdropColor(red: 0.18, green: 0.18, blue: 0.18)
}

/// Modes the on-screen control overlay can be in. The user picks one
/// from the inspector; clicking the preview surface in any mode other
/// than `.disabled` adds or removes a hint point.
enum OnScreenControlTool: Hashable, Sendable, CaseIterable, Identifiable {
    case disabled
    case foregroundHint
    case backgroundHint
    case eraseHint

    var id: Self { self }

    var displayName: String {
        switch self {
        case .disabled: return "Off"
        case .foregroundHint: return "Foreground"
        case .backgroundHint: return "Background"
        case .eraseHint: return "Erase"
        }
    }

    var systemImage: String {
        switch self {
        case .disabled: return "circle.slash"
        case .foregroundHint: return "plus.circle.fill"
        case .backgroundHint: return "minus.circle.fill"
        case .eraseHint: return "eraser"
        }
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
    /// Whether the transport is currently playing back. Drives the
    /// play / pause button glyph and the periodic playhead-advance
    /// loop.
    var isPlaying: Bool = false
    /// Whether playback should restart from frame 0 when it reaches
    /// the end of the clip. QuickTime's loop control mirrored.
    var loopEnabled: Bool = false
    /// Rolling average of the per-tick playback frame rate, in
    /// frames per second. Computed by `runPlaybackLoop` from the
    /// wall-clock interval between successive renders. Zero outside
    /// playback. Drives the green/orange fps badge in the transport
    /// bar so the user can see at a glance whether their machine is
    /// keeping up with realtime.
    var measuredPlaybackFPS: Double = 0
    /// The frame rate the loaded clip would play at if every render
    /// landed instantly — i.e. the source's nominal frame rate. The
    /// transport bar's fps badge compares `measuredPlaybackFPS`
    /// against this threshold to colour itself green vs orange.
    var targetPlaybackFPS: Double = 0
    /// Currently-active OSC tool — drives the click handler on the
    /// preview surface.
    var oscTool: OnScreenControlTool = .disabled
    /// Backdrop drawn behind the keyed preview image. Defaults to
    /// the transparency-aware checkerboard pattern; right-clicking
    /// the preview surfaces the picker.
    var previewBackdrop: PreviewBackdrop = .checkerboard
    /// Colour rendered behind the keyed image when `previewBackdrop`
    /// is `.customColor`. Persisted to UserDefaults so the user's
    /// last pick survives a launch.
    var customBackdropColor: BackdropColor = .default {
        didSet {
            guard customBackdropColor != oldValue else { return }
            BackdropPreferences.saveCustomColor(customBackdropColor)
        }
    }
    /// User-imported image rendered behind the keyed image when
    /// `previewBackdrop` is `.customImage`. The texture is built
    /// from the image data once at import time and re-used until
    /// the user picks a different image. SwiftUI invalidates on
    /// reassignment because the underlying object identity changes.
    var customBackdropTexture: (any MTLTexture)?
    /// Display name of the imported image (the URL's last path
    /// component) — surfaced in the menu so the user knows which
    /// file is currently in use.
    var customBackdropImageName: String?

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
    /// Background Task that drives playback. Replaces the legacy
    /// `Timer.scheduledTimer` loop so each tick can await the GPU
    /// render before advancing — earlier builds fired ticks at the
    /// source frame rate regardless of render latency, which on a
    /// short looping clip meant the displayed frame never actually
    /// caught up to the playhead.
    @ObservationIgnored
    private var playbackTask: Task<Void, Never>?

    init(renderEngine: StandaloneRenderEngine) {
        self.renderEngine = renderEngine
        self.customBackdropColor = BackdropPreferences.loadCustomColor() ?? .default
        // Restore the user's last imported image asynchronously so
        // the editor window appears immediately. If the bookmark is
        // stale (file moved / deleted) we silently fall back to
        // checkerboard rather than surfacing an error on launch.
        if let bookmarkData = BackdropPreferences.loadImageBookmark() {
            Task { [weak self, device = renderEngine.device] in
                await Self.restoreCustomBackdropImage(
                    bookmarkData: bookmarkData,
                    device: device,
                    into: self
                )
            }
        }
    }

    // MARK: - Custom backdrop import

    /// Loads `url` as the custom-image backdrop, switches the
    /// preview backdrop to `.customImage`, and persists a security-
    /// scoped bookmark so the same image is restored on next
    /// launch. Throws if the file is too large or the decode fails;
    /// callers surface the error in an alert.
    ///
    /// Decoding + texture upload runs on the main actor: `MTLTexture`
    /// isn't `Sendable`, so we'd have to wrap it in an unchecked
    /// transfer to push the work off main. For a typical backdrop
    /// image (PNG / JPEG ≤ 64 MB) the work is fast enough to land
    /// inside one frame; the import sheet is already modal so a few
    /// hundred milliseconds of UI freeze isn't perceptible.
    func importBackdropImage(from url: URL) async throws {
        let scopeStarted = url.startAccessingSecurityScopedResource()
        defer {
            if scopeStarted {
                url.stopAccessingSecurityScopedResource()
            }
        }
        let data = try Data(contentsOf: url)
        let bookmarkData = try url.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        let texture = try BackdropImageLoader.makeTexture(from: data, device: renderEngine.device)
        self.customBackdropTexture = texture
        self.customBackdropImageName = url.lastPathComponent
        BackdropPreferences.saveImageBookmark(bookmarkData)
        self.previewBackdrop = .customImage
        renderPreview(at: playheadTime)
    }

    /// Drops the imported image and falls back to the checkerboard
    /// backdrop. Clears the persisted bookmark too — next launch
    /// won't try to re-load a file the user just removed.
    func clearBackdropImage() {
        customBackdropTexture = nil
        customBackdropImageName = nil
        BackdropPreferences.clearImageBookmark()
        if previewBackdrop == .customImage {
            previewBackdrop = .checkerboard
            renderPreview(at: playheadTime)
        }
    }

    /// Resolves a security-scoped bookmark and pushes the resulting
    /// texture onto the view model. Runs on the main actor because
    /// `MTLTexture` isn't `Sendable`; the decode is fast enough not
    /// to disrupt the launch animation in practice (the editor
    /// window is already drawing by the time this fires).
    @MainActor
    private static func restoreCustomBackdropImage(
        bookmarkData: Data,
        device: any MTLDevice,
        into viewModel: EditorViewModel?
    ) async {
        var isStale = false
        let url: URL
        do {
            url = try URL(
                resolvingBookmarkData: bookmarkData,
                options: .withSecurityScope,
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
        } catch {
            // Bookmark unresolvable — likely the file was deleted
            // or moved. Drop the saved bookmark so we don't keep
            // trying.
            BackdropPreferences.clearImageBookmark()
            return
        }
        let scopeStarted = url.startAccessingSecurityScopedResource()
        defer {
            if scopeStarted {
                url.stopAccessingSecurityScopedResource()
            }
        }
        guard let data = try? Data(contentsOf: url) else { return }
        let texture: any MTLTexture
        do {
            texture = try BackdropImageLoader.makeTexture(from: data, device: device)
        } catch {
            return
        }
        guard let viewModel else { return }
        viewModel.customBackdropTexture = texture
        viewModel.customBackdropImageName = url.lastPathComponent
        if isStale {
            if let refreshed = try? url.bookmarkData(
                options: .withSecurityScope,
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            ) {
                BackdropPreferences.saveImageBookmark(refreshed)
            }
        }
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
        stopPlayback()
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

    // MARK: - Playback transport

    /// Starts or stops playback. Each tick of the playback timer
    /// advances the playhead by one source frame; when it reaches
    /// the end of the clip we either loop back to frame 0 or stop
    /// based on `loopEnabled`.
    func togglePlayback() {
        if isPlaying {
            stopPlayback()
        } else {
            startPlayback()
        }
    }

    private func startPlayback() {
        guard let clipInfo, !isPlaying else { return }
        targetPlaybackFPS = max(Double(clipInfo.nominalFrameRate), 0)
        measuredPlaybackFPS = 0
        isPlaying = true
        playbackTask?.cancel()
        playbackTask = Task { @MainActor [weak self] in
            await self?.runPlaybackLoop()
        }
    }

    private func stopPlayback() {
        playbackTask?.cancel()
        playbackTask = nil
        isPlaying = false
        measuredPlaybackFPS = 0
    }

    /// Awaits-each-frame playback loop. Each iteration:
    ///
    /// 1. Picks the next frame index (wrapping when `loopEnabled`).
    /// 2. Moves the playhead so the slider/time label track in real
    ///    time as the user expects.
    /// 3. **Awaits** `performRender(at:)` for that frame so the
    ///    user actually sees every frame land — the legacy timer-
    ///    based path fired ticks at the nominal frame rate even
    ///    when the GPU couldn't keep up, which on a short looping
    ///    clip silently dropped most renders mid-flight.
    /// 4. If the render finished faster than the source's per-frame
    ///    interval, sleeps the remainder so playback runs at the
    ///    real frame rate instead of turbo-mode.
    /// 5. Updates `measuredPlaybackFPS` from the wall-clock interval
    ///    so the transport bar's badge can colour itself green
    ///    (≥ 95% of target) or orange (below).
    private func runPlaybackLoop() async {
        guard let clipInfo else { return }
        let frameRate = max(Double(clipInfo.nominalFrameRate), 0.001)
        let frameIntervalSeconds = 1.0 / frameRate
        // Window of recent samples used to smooth the displayed fps
        // — without smoothing, a single slow frame would slam the
        // badge into the orange band even when the average is fine.
        // Sized to ~1 s of history so the indicator settles within
        // a wall-clock second of any change.
        let fpsSampleCap = max(Int(frameRate.rounded()), 1)
        var fpsSamples: [Double] = []

        while isPlaying && !Task.isCancelled {
            let tickStart = Date()

            let currentFrame = currentFrameIndex(at: playheadTime)
            let lastFrame = max(totalFrames - 1, 0)
            let nextFrame: Int
            if currentFrame >= lastFrame {
                if loopEnabled {
                    nextFrame = 0
                } else {
                    isPlaying = false
                    break
                }
            } else {
                nextFrame = currentFrame + 1
            }

            let target = CMTime(
                seconds: Double(nextFrame) / frameRate,
                preferredTimescale: clipInfo.timescale
            )
            playheadTime = target

            do {
                try await performRender(at: target)
            } catch {
                if !Task.isCancelled {
                    self.phase = .loadFailed(error.localizedDescription)
                }
                isPlaying = false
                break
            }
            if Task.isCancelled { break }

            let renderSeconds = Date().timeIntervalSince(tickStart)
            if renderSeconds < frameIntervalSeconds {
                let throttle = frameIntervalSeconds - renderSeconds
                try? await Task.sleep(for: .seconds(throttle))
            }
            if Task.isCancelled { break }

            let tickSeconds = Date().timeIntervalSince(tickStart)
            let achievedFPS = 1.0 / max(tickSeconds, 0.001)
            fpsSamples.append(achievedFPS)
            while fpsSamples.count > fpsSampleCap {
                fpsSamples.removeFirst()
            }
            measuredPlaybackFPS = fpsSamples.reduce(0, +) / Double(fpsSamples.count)
        }
        isPlaying = false
        measuredPlaybackFPS = 0
    }

    /// Awaitable variant of `renderPreview(at:)` used by the
    /// playback loop so each tick can wait for the rendered frame
    /// to actually land before advancing. Throws on render or
    /// decode failure so the caller can stop playback cleanly. Does
    /// not interact with `pendingPreviewTask` — that field exists
    /// to coalesce parameter-edit re-renders, which playback never
    /// fights with because the playback task owns the rendering
    /// schedule end-to-end.
    @MainActor
    private func performRender(at time: CMTime) async throws {
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
        let pixelBuffer = try await source.makeFrame(atTime: frameTime)
        let result = try renderEngine.render(
            source: pixelBuffer,
            state: stateForFrame,
            renderTime: frameTime
        )
        self.latestPreview = PreviewFrame(
            texture: result.destinationTexture,
            pixelBuffer: result.destinationPixelBuffer,
            presentationTime: frameTime
        )
        self.renderBackendDescription = result.report.backendDescription
        self.lastEngineDescription = result.report.guideSourceDescription
    }

    /// Toggles the loop affordance. Keeps playback running if it
    /// was; lets the user enable it mid-playback so the transport
    /// keeps the playhead from running off the end.
    func toggleLoop() {
        loopEnabled.toggle()
    }

    private func seek(toFrameIndex frameIndex: Int) {
        guard let clipInfo else { return }
        let frameRate = max(Double(clipInfo.nominalFrameRate), 0.001)
        let target = CMTime(
            seconds: Double(frameIndex) / frameRate,
            preferredTimescale: clipInfo.timescale
        )
        playheadTime = target
        renderPreview(at: target)
    }

    // MARK: - Transport

    /// Moves the playhead to a normalised position (0…1) interpreted
    /// as a position within the clip's **frame range**. Earlier
    /// builds mapped to position-within-duration, which on a short
    /// clip (the asset's duration is one frame past the last sample)
    /// meant the user couldn't drag the slider all the way to the
    /// last frame — a 4-frame clip's slider stopped at 0.75, hiding
    /// the final sample behind the right edge.
    ///
    /// Frame-index space: 0 → frame 0, 1 → frame N-1.
    func scrub(toNormalized fraction: Double) {
        guard let clipInfo else { return }
        let clamped = min(max(fraction, 0), 1)
        let lastFrameIndex = max(totalFrames - 1, 0)
        let frameRate = max(Double(clipInfo.nominalFrameRate), 0.001)
        let targetFrame = Int((clamped * Double(lastFrameIndex)).rounded())
        let target = CMTime(
            seconds: Double(targetFrame) / frameRate,
            preferredTimescale: clipInfo.timescale
        )
        playheadTime = target
        renderPreview(at: target)
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

    // MARK: - On-screen control (OSC)

    /// Handles a click anywhere on the preview surface. Coordinates
    /// are normalised (0…1) — `(0, 0)` is top-left, `(1, 1)` is
    /// bottom-right of the rendered frame, matching the convention
    /// the FxPlug OSC writes into the `HintPointSet` parameter.
    /// No-op when the tool is `.disabled`.
    func handleOSCClick(atNormalizedPoint point: CGPoint) {
        let tool = oscTool
        let clampedX = min(max(Double(point.x), 0), 1)
        let clampedY = min(max(Double(point.y), 0), 1)
        switch tool {
        case .disabled:
            return
        case .foregroundHint:
            state.hintPointSet.add(HintPoint(x: clampedX, y: clampedY, kind: .foreground))
        case .backgroundHint:
            state.hintPointSet.add(HintPoint(x: clampedX, y: clampedY, kind: .background))
        case .eraseHint:
            state.hintPointSet.removeNearest(toX: clampedX, y: clampedY, tolerance: 0.05)
        }
        // Hints feed the MLX bridge at analysis time. Re-render the
        // preview now so the dot appears immediately, and prompt the
        // user to re-analyse to see the matte respond to the new
        // hints.
        parameterDidChange()
    }

    /// Wipes every placed hint point.
    func clearAllHints() {
        state.hintPointSet.clear()
        parameterDidChange()
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
        stopPlayback()
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

/// Disk-backed persistence for the custom backdrop colour and
/// imported image. Sits next to the view model rather than in a
/// dedicated file because it has exactly one consumer; if a
/// second consumer ever appears, lift it into its own file.
enum BackdropPreferences {
    static let customColorKey = "CorridorKey.PreviewBackdrop.CustomColor"
    static let imageBookmarkKey = "CorridorKey.PreviewBackdrop.ImageBookmark"

    static func loadCustomColor() -> BackdropColor? {
        guard let data = UserDefaults.standard.data(forKey: customColorKey) else {
            return nil
        }
        return try? JSONDecoder().decode(BackdropColor.self, from: data)
    }

    static func saveCustomColor(_ color: BackdropColor) {
        guard let data = try? JSONEncoder().encode(color) else { return }
        UserDefaults.standard.set(data, forKey: customColorKey)
    }

    static func loadImageBookmark() -> Data? {
        UserDefaults.standard.data(forKey: imageBookmarkKey)
    }

    static func saveImageBookmark(_ data: Data) {
        UserDefaults.standard.set(data, forKey: imageBookmarkKey)
    }

    static func clearImageBookmark() {
        UserDefaults.standard.removeObject(forKey: imageBookmarkKey)
    }
}
