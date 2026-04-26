//
//  ProResExporter.swift
//  Corridor Key Toolbox — Standalone Editor
//
//  Streams every frame of the source clip through Corridor Key, then
//  encodes the result as Apple ProRes 4444 (with optional alpha) into
//  a QuickTime container via `AVAssetWriter`. Mirrors the FxPlug
//  render path one frame at a time; cached mattes from the analyse
//  pass let the exporter skip MLX entirely so the export wall-time
//  scales with the post-process chain alone (~5–15 ms / 4K frame on
//  M-series GPUs).
//
//  Output dimensions match the clip's natural render size after the
//  preferred-track-transform rotation, so portrait clips export the
//  right way up.
//

import Foundation
import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox

/// User-visible export options. Surfaced by the export sheet.
struct ExportOptions: Sendable, Equatable {
    /// Destination URL chosen via `NSSavePanel`. Always inside a
    /// user-selected location so the wrapper app's sandbox can write
    /// to it.
    let destination: URL
    /// Pixel format chosen for ProRes encoding. ProRes 4444 supports
    /// alpha; ProRes 422 HQ does not but produces a smaller file.
    let codec: ProResCodec
    /// When true, embeds the matte in ProRes 4444's alpha channel.
    /// Ignored for ProRes 422 (no alpha channel).
    let preserveAlpha: Bool

    enum ProResCodec: String, Sendable, CaseIterable, Identifiable {
        case proRes4444
        case proRes422HQ
        case proRes422

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .proRes4444: return "Apple ProRes 4444"
            case .proRes422HQ: return "Apple ProRes 422 HQ"
            case .proRes422: return "Apple ProRes 422"
            }
        }

        /// AV Foundation's codec identifier matching this case.
        var avFoundationCodec: AVVideoCodecType {
            switch self {
            case .proRes4444: return .proRes4444
            case .proRes422HQ: return .proRes422HQ
            case .proRes422: return .proRes422
            }
        }

        /// True when this codec carries an alpha channel.
        var supportsAlpha: Bool {
            switch self {
            case .proRes4444: return true
            case .proRes422HQ, .proRes422: return false
            }
        }
    }
}

/// Status events emitted while the export pass runs. The export sheet
/// listens to drive its progress UI.
enum ExportRunnerEvent: @unchecked Sendable {
    case started(totalFrames: Int)
    case frameWritten(frameIndex: Int)
    case completed(URL)
    case failed(String)
    case cancelled
}

/// Runs the Corridor-Key-then-ProRes pipeline for one clip. Owns the
/// reader, writer, and pixel-buffer pool used by AVAssetWriter; held
/// for the duration of an export and dropped on completion.
actor ProResExporter {

    private let renderEngine: StandaloneRenderEngine
    private let videoSource: VideoSource
    private let project: ExportProjectSnapshot

    init(
        renderEngine: StandaloneRenderEngine,
        videoSource: VideoSource,
        project: ExportProjectSnapshot
    ) {
        self.renderEngine = renderEngine
        self.videoSource = videoSource
        self.project = project
    }

    /// Executes the export. Throws on writer setup failure; per-frame
    /// errors are reported via `eventHandler` so the UI can surface
    /// them without unwinding the whole task.
    func run(
        options: ExportOptions,
        eventHandler: @Sendable (ExportRunnerEvent) -> Void
    ) async {
        let info = videoSource.info
        let frameCount = videoSource.totalFrameCount()
        let renderSize = info.renderSize
        let frameRate = max(Double(info.nominalFrameRate), 0.001)
        let timescale = info.timescale

        // Tear down any previous file at the destination — AVAssetWriter
        // refuses to overwrite, and the user already confirmed via the
        // save sheet that this URL is OK to replace.
        if FileManager.default.fileExists(atPath: options.destination.path) {
            do {
                try FileManager.default.removeItem(at: options.destination)
            } catch {
                eventHandler(.failed("Couldn't remove existing file: \(error.localizedDescription)"))
                return
            }
        }

        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: options.destination, fileType: .mov)
        } catch {
            eventHandler(.failed("Couldn't create writer: \(error.localizedDescription)"))
            return
        }

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: options.codec.avFoundationCodec.rawValue,
            AVVideoWidthKey: Int(renderSize.width.rounded()),
            AVVideoHeightKey: Int(renderSize.height.rounded()),
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ]
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput.expectsMediaDataInRealTime = false

        let pixelBufferAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf,
            kCVPixelBufferWidthKey as String: Int(renderSize.width.rounded()),
            kCVPixelBufferHeightKey as String: Int(renderSize.height.rounded()),
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: pixelBufferAttrs
        )
        guard writer.canAdd(videoInput) else {
            eventHandler(.failed("AVAssetWriter rejected the video input."))
            return
        }
        writer.add(videoInput)

        guard writer.startWriting() else {
            eventHandler(.failed(writer.error?.localizedDescription ?? "Writer failed to start."))
            return
        }
        writer.startSession(atSourceTime: .zero)
        eventHandler(.started(totalFrames: frameCount))

        let frameReader: VideoFrameReader
        do {
            frameReader = try videoSource.makeFrameReader(startTime: .zero)
        } catch {
            writer.cancelWriting()
            eventHandler(.failed("Couldn't open clip for export: \(error.localizedDescription)"))
            return
        }
        defer { frameReader.cancel() }

        var frameIndex = 0
        while !Task.isCancelled {
            let frame: ReaderFrame?
            do {
                frame = try frameReader.nextFrame()
            } catch {
                writer.cancelWriting()
                eventHandler(.failed("Read failed at frame \(frameIndex): \(error.localizedDescription)"))
                return
            }
            guard let frame else { break }

            // Wait until the writer is ready to accept more frames.
            // AVAssetWriter buffers a bounded number internally; if we
            // push too quickly it stalls instead of dropping.
            while !videoInput.isReadyForMoreMediaData {
                if Task.isCancelled { break }
                try? await Task.sleep(for: .milliseconds(2))
            }
            if Task.isCancelled { break }

            let renderTime: CMTime
            if frame.presentationTime.isValid && frame.presentationTime.isNumeric {
                renderTime = frame.presentationTime
            } else {
                renderTime = CMTime(
                    seconds: Double(frameIndex) / frameRate,
                    preferredTimescale: timescale
                )
            }

            // Resolve cached matte for this frame, if analysis has run.
            // Without one, the renderer pass-throughs and the export
            // emits the source untouched — which is the explicit "not
            // analysed yet" state we surface in the FxPlug too.
            var stateForFrame = project.state
            if let cached = project.cachedMatte(forFrameIndex: frameIndex) {
                stateForFrame.cachedMatteBlob = cached.blob
                stateForFrame.cachedMatteInferenceResolution = cached.inferenceResolution
            } else {
                stateForFrame.cachedMatteBlob = nil
                stateForFrame.cachedMatteInferenceResolution = 0
            }
            stateForFrame.destinationLongEdgePixels = max(
                Int(renderSize.width.rounded()),
                Int(renderSize.height.rounded())
            )

            let result: StandaloneRenderResult
            do {
                result = try renderEngine.render(
                    source: frame.pixelBuffer,
                    state: stateForFrame,
                    renderTime: renderTime
                )
            } catch {
                writer.cancelWriting()
                eventHandler(.failed("Render failed at frame \(frameIndex): \(error.localizedDescription)"))
                return
            }

            if !adaptor.append(result.destinationPixelBuffer, withPresentationTime: renderTime) {
                let writerError = writer.error?.localizedDescription ?? "unknown writer failure"
                writer.cancelWriting()
                eventHandler(.failed("Write failed at frame \(frameIndex): \(writerError)"))
                return
            }
            eventHandler(.frameWritten(frameIndex: frameIndex))
            frameIndex += 1
            await Task.yield()
        }

        if Task.isCancelled {
            writer.cancelWriting()
            eventHandler(.cancelled)
            return
        }

        videoInput.markAsFinished()
        await writer.finishWriting()
        if writer.status == .completed {
            eventHandler(.completed(options.destination))
        } else if let error = writer.error {
            eventHandler(.failed(error.localizedDescription))
        } else {
            eventHandler(.failed("Writer finished with status \(writer.status.rawValue)."))
        }
    }
}

/// Read-only snapshot of the project handed to the exporter. Contains
/// the parameters and the matte cache the export needs; isolated from
/// `EditorProject` so the exporter does not race the live editor state.
struct ExportProjectSnapshot: @unchecked Sendable {
    let state: PluginStateData
    private let cachedMattes: [Int: MatteCacheEntry]

    init(state: PluginStateData, cachedMattes: [Int: MatteCacheEntry]) {
        self.state = state
        self.cachedMattes = cachedMattes
    }

    func cachedMatte(forFrameIndex index: Int) -> MatteCacheEntry? {
        cachedMattes[index]
    }
}
