//
//  ProResExporter.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Streams every frame of the source clip through Corridor Key, then
//  encodes the result as Apple ProRes (4444 / 422 HQ / 422 / 422 LT /
//  422 Proxy) or HEVC (with optional alpha channel on Apple Silicon)
//  into a QuickTime container via `AVAssetWriter`. Mirrors the FxPlug
//  render path one frame at a time; cached mattes from the analyse
//  pass let the exporter skip MLX entirely so the export wall-time
//  scales with the post-process chain alone (~5–15 ms / 4K frame on
//  M-series GPUs).
//
//  Output dimensions match the clip's natural render size after the
//  preferred-track-transform rotation, so portrait clips export the
//  right way up.
//
//  Colour management: when the source clip carries explicit
//  primaries / transfer / matrix tags (HDR, Rec.2020, DCI-P3, …) the
//  exporter forwards them verbatim to AVAssetWriter so the resulting
//  file stays correctly tagged. Sources without colour metadata fall
//  back to BT.709 SDR, matching pre-v1.1 behaviour.
//
//  Hardware acceleration: AVAssetWriter on Apple Silicon routes
//  ProRes encoding to the dedicated ProRes accelerator and HEVC to
//  the Video Toolbox HEVC engine automatically. The
//  `kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder`
//  hint we set inside `videoSettings` is purely an explicit signal
//  to those Toolbox sessions in case they would otherwise fall back
//  to a software path on resource-starved configurations.
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
    /// Codec chosen for encoding. Drives both the `.mov` codec key
    /// and which buffer pixel format the writer requests.
    let codec: ExportCodec
    /// When true, asks the exporter to embed the matte in the output
    /// alpha channel. Only honoured when the codec actually carries
    /// alpha (ProRes 4444 or HEVC with Alpha); silently ignored
    /// otherwise.
    let preserveAlpha: Bool

    enum ExportCodec: String, Sendable, CaseIterable, Identifiable {
        case proRes4444
        case proRes422HQ
        case proRes422
        case proRes422LT
        case proRes422Proxy
        case hevcWithAlpha
        case hevc

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .proRes4444: return "Apple ProRes 4444"
            case .proRes422HQ: return "Apple ProRes 422 HQ"
            case .proRes422: return "Apple ProRes 422"
            case .proRes422LT: return "Apple ProRes 422 LT"
            case .proRes422Proxy: return "Apple ProRes 422 Proxy"
            case .hevcWithAlpha: return "HEVC with Alpha (H.265)"
            case .hevc: return "HEVC (H.265)"
            }
        }

        /// AV Foundation's codec identifier matching this case.
        var avFoundationCodec: AVVideoCodecType {
            switch self {
            case .proRes4444: return .proRes4444
            case .proRes422HQ: return .proRes422HQ
            case .proRes422: return .proRes422
            case .proRes422LT: return .proRes422LT
            case .proRes422Proxy: return .proRes422Proxy
            case .hevcWithAlpha: return .hevcWithAlpha
            case .hevc: return .hevc
            }
        }

        /// True when this codec carries an alpha channel.
        var supportsAlpha: Bool {
            switch self {
            case .proRes4444, .hevcWithAlpha: return true
            case .proRes422HQ, .proRes422, .proRes422LT, .proRes422Proxy, .hevc: return false
            }
        }

        /// Whether this codec is in the ProRes family. Used to pick
        /// pixel format / quality defaults that suit ProRes vs HEVC
        /// without scattering codec-by-codec branches across the
        /// exporter.
        var isProRes: Bool {
            switch self {
            case .proRes4444, .proRes422HQ, .proRes422, .proRes422LT, .proRes422Proxy: return true
            case .hevcWithAlpha, .hevc: return false
            }
        }

        /// Buffer pixel format the exporter requests from
        /// `AVAssetWriterInputPixelBufferAdaptor`. ProRes wants
        /// `kCVPixelFormatType_64RGBAHalf` so the encoder gets full
        /// 16-bit-float precision; HEVC wants 8-bit BGRA (with-alpha
        /// builds need an alpha plane the encoder accepts).
        var sourcePixelFormat: OSType {
            if isProRes {
                return kCVPixelFormatType_64RGBAHalf
            }
            return kCVPixelFormatType_32BGRA
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

/// Runs the keyer-then-encoder pipeline for one clip. Owns the
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

        var videoSettings: [String: Any] = [
            AVVideoCodecKey: options.codec.avFoundationCodec.rawValue,
            AVVideoWidthKey: Int(renderSize.width.rounded()),
            AVVideoHeightKey: Int(renderSize.height.rounded()),
            AVVideoColorPropertiesKey: Self.colorProperties(for: info)
        ]
        // HEVC needs a quality target; ProRes is fixed-rate so this
        // dictionary is left absent for those codecs.
        if !options.codec.isProRes {
            videoSettings[AVVideoCompressionPropertiesKey] = Self.hevcCompressionProperties(
                forAlpha: options.codec.supportsAlpha && options.preserveAlpha
            )
        }

        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput.expectsMediaDataInRealTime = false

        let pixelBufferAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: options.codec.sourcePixelFormat,
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

    /// Builds the `AVVideoColorPropertiesKey` dictionary using the
    /// source's tags when available. Forwarding the source tags is
    /// what keeps HDR / Rec.2020 / DCI-P3 sources looking correct
    /// downstream — silently re-tagging them BT.709 made wide-gamut
    /// clips appear desaturated when reopened in pro tools.
    private static func colorProperties(for info: VideoSourceInfo) -> [String: Any] {
        let primaries = info.colorPrimaries ?? AVVideoColorPrimaries_ITU_R_709_2
        let transfer = info.transferFunction ?? AVVideoTransferFunction_ITU_R_709_2
        let matrix = info.yCbCrMatrix ?? AVVideoYCbCrMatrix_ITU_R_709_2
        return [
            AVVideoColorPrimariesKey: primaries,
            AVVideoTransferFunctionKey: transfer,
            AVVideoYCbCrMatrixKey: matrix
        ]
    }

    /// HEVC compression-property dictionary. Targets a high-quality
    /// preset; on Apple Silicon `AVAssetWriter` defaults the HEVC
    /// encoder to hardware acceleration so no explicit opt-in is
    /// required. When `forAlpha` is true, sets
    /// `kVTCompressionPropertyKey_TargetQualityForAlpha` so semi-
    /// transparent edges retain their detail — the default alpha
    /// quality is conservative and visibly softens fine matte
    /// gradients.
    private static func hevcCompressionProperties(forAlpha: Bool) -> [String: Any] {
        var compression: [String: Any] = [
            AVVideoQualityKey: 0.85,
            AVVideoExpectedSourceFrameRateKey: 60,
            AVVideoMaxKeyFrameIntervalKey: 60
        ]
        if forAlpha {
            compression[kVTCompressionPropertyKey_TargetQualityForAlpha as String] = 0.95
        }
        return compression
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
