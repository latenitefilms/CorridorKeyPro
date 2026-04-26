//
//  VideoSource.swift
//  Corridor Key Toolbox — Standalone Editor
//
//  Wraps an `AVURLAsset` so the standalone editor can:
//
//  * Inspect the clip's natural size, frame rate, duration, and
//    transform (used to build the right preview / output dimensions).
//  * Pull a specific frame into a Metal-compatible `CVPixelBuffer` via
//    `AVAssetImageGenerator` (used for scrubbing and analyse).
//  * Stream every frame in display order via an `AsyncThrowingStream`
//    of `(presentationTime, pixelBuffer)` (used for the export pass).
//
//  Both code paths request the same pixel format — the renderer's
//  preferred RGBA16Float — so the GPU stages do not need to handle
//  format-conversion edge cases on the hot path. Anything the host
//  cannot decode at that format falls back to BGRA8Unorm and the
//  pipeline applies the matching shader path.
//

import Foundation
import AVFoundation
import CoreImage
import CoreVideo
import CoreMedia

/// Static information about a video clip — the bits the standalone
/// editor needs to render the correct preview and lay out the timeline.
struct VideoSourceInfo: Sendable {
    let url: URL
    /// Source-pixel resolution after applying any `preferredTransform`
    /// rotation. The renderer always sees the clip the right way up.
    let renderSize: CGSize
    /// Display frame rate. Used to drive the transport and to sample
    /// frames at the natural interval during analyse / export.
    let nominalFrameRate: Float
    /// Total clip duration. Cached so the UI does not have to wait
    /// on `AVAsset.load(.duration)` per frame.
    let duration: CMTime
    /// Native pixel format of the source track's first sample. Used to
    /// pick a matching destination pixel format on export — e.g. emit
    /// ProRes 4444 at 16 bpc when the source is HDR / float.
    let nativePixelFormat: OSType
    /// Time scale of the source track. Used to build CMTimes that align
    /// to the source frame grid (so scrubbing snaps to whole frames).
    let timescale: CMTimeScale
    /// AV Foundation track identifier for the video. Cached so
    /// downstream readers reference the same track.
    let videoTrackID: CMPersistentTrackID
    /// Preferred transform of the source track (rotation matrix).
    /// Applied automatically by `AVAssetImageGenerator` and the export
    /// reader when `appliesPreferredTrackTransform` is true.
    let preferredTransform: CGAffineTransform
}

enum VideoSourceError: Error, CustomStringConvertible {
    case noVideoTrack(URL)
    case readerSetupFailed(any Error)
    case readerOutputUnavailable
    case sampleBufferUnavailable
    case pixelBufferUnavailable

    var description: String {
        switch self {
        case .noVideoTrack(let url):
            return "Corridor Key Toolbox couldn't find a video track in \(url.lastPathComponent)."
        case .readerSetupFailed(let error):
            return "Corridor Key Toolbox couldn't open the clip for reading: \(error.localizedDescription)"
        case .readerOutputUnavailable:
            return "Corridor Key Toolbox couldn't attach a video output to the clip reader."
        case .sampleBufferUnavailable:
            return "Corridor Key Toolbox couldn't read the next sample buffer from the clip."
        case .pixelBufferUnavailable:
            return "Corridor Key Toolbox decoded a sample but no pixel data came with it."
        }
    }
}

/// Loads metadata for a clip and exposes scrub + stream APIs over its
/// pixel data. Holds onto the `AVURLAsset` for the lifetime of the
/// editor session so reader objects can be recreated cheaply.
///
/// Marked `@unchecked Sendable` because `AVURLAsset` is documented as
/// thread-safe and the rest of this class's state is immutable after
/// `init`. Using a regular class instead of an actor lets pipeline
/// hand-offs (e.g. handing a freshly-decoded `CVPixelBuffer` to the
/// main actor) avoid Swift 6's non-Sendable-CoreFoundation friction.
///
/// **Sandbox handling.** Files chosen via `NSOpenPanel` /
/// `.fileImporter` come back as security-scoped URLs. The wrapper app
/// is sandboxed, so opening the underlying file requires bracketing
/// access with `startAccessingSecurityScopedResource()` /
/// `stopAccessingSecurityScopedResource()`. We start the access in
/// `init` and hold it for the lifetime of this `VideoSource` so all
/// downstream `AVURLAsset` reads — frame generation, frame readers,
/// metadata loading — succeed even though the user only granted
/// access once at the file picker. URLs that aren't security-scoped
/// (e.g. files inside `~/Movies` accessed via the
/// `movies.read-write` entitlement) return `false` from
/// `startAccessing…`; we still record that and skip the matching
/// `stop…` call so we never unbalance the access count.
final class VideoSource: @unchecked Sendable {

    let info: VideoSourceInfo
    private let asset: AVURLAsset
    private let url: URL
    private let didStartAccessing: Bool

    /// Loads the clip at `url` asynchronously, populating `info`. Throws
    /// if the asset has no video track or AV Foundation can't decode
    /// the headers.
    init(url: URL) async throws {
        // Security-scoped access must be acquired BEFORE handing the
        // URL to AVURLAsset — the asset reads the underlying file to
        // parse headers, and the sandbox refuses without an active
        // bracket. Returns `false` for non-security-scoped URLs (e.g.
        // a file inside `~/Movies` accessed via the entitlement) so
        // we know not to call `stop…` later.
        self.url = url
        self.didStartAccessing = url.startAccessingSecurityScopedResource()
        let options: [String: Any] = [
            AVURLAssetPreferPreciseDurationAndTimingKey: true
        ]
        let asset = AVURLAsset(url: url, options: options)
        self.asset = asset
        do {
            self.info = try await Self.loadInfo(for: asset, url: url)
        } catch {
            // Failed before assigning to `self.info` — release the
            // access bracket immediately because `deinit` won't fire
            // for a partially-initialised instance.
            if didStartAccessing {
                url.stopAccessingSecurityScopedResource()
            }
            throw error
        }
    }

    deinit {
        if didStartAccessing {
            url.stopAccessingSecurityScopedResource()
        }
    }

    private static func loadInfo(
        for asset: AVURLAsset,
        url: URL
    ) async throws -> VideoSourceInfo {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = tracks.first else {
            throw VideoSourceError.noVideoTrack(url)
        }
        // Swift 6 strict concurrency: `AVAssetTrack` is not `Sendable`,
        // so we can't `async let` its loaders concurrently — that would
        // imply sending the track across actor boundaries. Sequence the
        // loads instead. Header parsing dominates the wall time anyway,
        // and the second-and-later calls are served from the asset's
        // internal property cache so the perf cost is negligible.
        let naturalSize = try await videoTrack.load(.naturalSize)
        let preferredTransform = try await videoTrack.load(.preferredTransform)
        let nominalFrameRate = try await videoTrack.load(.nominalFrameRate)
        let formatDescriptions = try await videoTrack.load(.formatDescriptions)
        let duration = try await asset.load(.duration)
        let timescale = try await videoTrack.load(.naturalTimeScale)
        // `trackID` is a synchronous, non-loadable property on
        // `AVAssetTrack`. Reading it directly here avoids the
        // `AVAsyncProperty` lookup which doesn't expose this field.
        let trackID = videoTrack.trackID

        // Apply the preferred transform to natural size so renderSize
        // already accounts for portrait clips. AVAssetImageGenerator and
        // AVAssetReader (with appliesPreferredTrackTransform = true)
        // both rotate frames in flight; lining renderSize up with that
        // means the renderer always sees the clip right-way-up.
        let rotated = naturalSize.applying(preferredTransform)
        let renderSize = CGSize(width: abs(rotated.width), height: abs(rotated.height))

        // First format description's pixel format hints at the source
        // bit depth. Used by ExportSession to pick a matching output
        // pixel format for ProRes.
        let nativePixelFormat: OSType
        if let first = formatDescriptions.first {
            nativePixelFormat = CMFormatDescriptionGetMediaSubType(first)
        } else {
            nativePixelFormat = kCVPixelFormatType_64RGBAHalf
        }

        return VideoSourceInfo(
            url: url,
            renderSize: renderSize,
            nominalFrameRate: nominalFrameRate,
            duration: duration,
            nativePixelFormat: nativePixelFormat,
            timescale: timescale,
            videoTrackID: trackID,
            preferredTransform: preferredTransform
        )
    }

    /// Snaps `time` to the nearest source frame. Used by the transport
    /// so scrubbing rests on whole frames, never between them.
    func nearestFrameTime(to time: CMTime) -> CMTime {
        let fps = max(Double(info.nominalFrameRate), 0.001)
        let frameIndex = (time.seconds * fps).rounded()
        return CMTime(seconds: frameIndex / fps, preferredTimescale: info.timescale)
    }

    /// Returns the total whole-frame count in the clip. The transport
    /// uses this to size the scrubber.
    func totalFrameCount() -> Int {
        let fps = Double(info.nominalFrameRate)
        guard fps > 0 else { return 1 }
        return max(1, Int((info.duration.seconds * fps).rounded()))
    }

    /// Pulls a single frame at `time` as a Metal-compatible
    /// `CVPixelBuffer`. Used to drive the preview while the user is
    /// scrubbing or analysing — both want random access.
    ///
    /// `preferredPixelFormat` controls the requested decode format. The
    /// renderer is happy with RGBA16Float (default) or BGRA8Unorm; the
    /// generator may downgrade if the source can't produce the
    /// requested format.
    func makeFrame(
        atTime time: CMTime,
        preferredPixelFormat: OSType = kCVPixelFormatType_64RGBAHalf
    ) async throws -> CVPixelBuffer {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = info.renderSize

        // Image generation in the Cocoa-friendly API only emits CGImage,
        // which requires a CPU detour. We request a CVPixelBuffer
        // directly via the lower-level callback API for zero-copy
        // hand-off to Metal.
        return try await withCheckedThrowingContinuation { continuation in
            generator.generateCGImageAsynchronously(for: time) { cgImage, _, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let cgImage else {
                    continuation.resume(throwing: VideoSourceError.pixelBufferUnavailable)
                    return
                }
                do {
                    let pixelBuffer = try Self.makePixelBuffer(
                        from: cgImage,
                        size: self.info.renderSize,
                        pixelFormat: preferredPixelFormat
                    )
                    continuation.resume(returning: pixelBuffer)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Builds an `AVAssetReader` that streams every frame of the clip
    /// in display order. Returned reader is started; callers pull
    /// sample buffers off `output` and convert them via `pixelBuffer`.
    /// Use this for the export pass — it is markedly faster than
    /// repeated `AVAssetImageGenerator` calls.
    func makeFrameReader(
        startTime: CMTime = .zero,
        preferredPixelFormat: OSType = kCVPixelFormatType_64RGBAHalf
    ) throws -> VideoFrameReader {
        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw VideoSourceError.readerSetupFailed(error)
        }
        reader.timeRange = CMTimeRange(start: startTime, end: info.duration)
        guard let videoTrack = asset.tracks(withMediaType: .video).first else {
            throw VideoSourceError.noVideoTrack(info.url)
        }
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: preferredPixelFormat,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw VideoSourceError.readerOutputUnavailable
        }
        reader.add(output)
        guard reader.startReading() else {
            throw VideoSourceError.readerSetupFailed(
                reader.error ?? VideoSourceError.readerOutputUnavailable
            )
        }
        return VideoFrameReader(
            reader: reader,
            output: output,
            preferredTransform: info.preferredTransform
        )
    }

    // MARK: - Helpers

    private static func makePixelBuffer(
        from cgImage: CGImage,
        size: CGSize,
        pixelFormat: OSType
    ) throws -> CVPixelBuffer {
        let width = Int(size.width.rounded())
        let height = Int(size.height.rounded())
        let pixelBuffer = try PixelBufferTextureBridge.makeMetalCompatiblePixelBuffer(
            width: width,
            height: height,
            pixelFormat: pixelFormat
        )
        // Render the CGImage into the buffer via Core Image so we get
        // colour-managed wide-gamut conversion for free. Core Image
        // picks the right colour space based on the buffer's IOSurface
        // metadata; we leave that nil so the system default applies.
        let context = CIContext(options: [.useSoftwareRenderer: false])
        let inputImage = CIImage(cgImage: cgImage)
        let scale = CGAffineTransform(
            scaleX: CGFloat(width) / inputImage.extent.width,
            y: CGFloat(height) / inputImage.extent.height
        )
        context.render(
            inputImage.transformed(by: scale),
            to: pixelBuffer,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            colorSpace: CGColorSpace(name: CGColorSpace.extendedSRGB)
        )
        return pixelBuffer
    }
}

/// Wraps an `AVAssetReader` plus its track output so callers don't
/// have to juggle the pair. Used by the export pass to pull frames in
/// display order.
final class VideoFrameReader: @unchecked Sendable {
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    let preferredTransform: CGAffineTransform

    init(
        reader: AVAssetReader,
        output: AVAssetReaderTrackOutput,
        preferredTransform: CGAffineTransform
    ) {
        self.reader = reader
        self.output = output
        self.preferredTransform = preferredTransform
    }

    /// Blocks the caller until the next frame is decoded. Returns
    /// `nil` when the reader runs out of data. Throws on reader errors.
    func nextFrame() throws -> ReaderFrame? {
        guard let sample = output.copyNextSampleBuffer() else {
            switch reader.status {
            case .completed: return nil
            case .failed:
                throw reader.error ?? VideoSourceError.sampleBufferUnavailable
            case .cancelled:
                return nil
            default:
                return nil
            }
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
            throw VideoSourceError.pixelBufferUnavailable
        }
        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sample)
        let duration = CMSampleBufferGetDuration(sample)
        return ReaderFrame(
            pixelBuffer: pixelBuffer,
            presentationTime: presentationTime,
            duration: duration
        )
    }

    func cancel() {
        reader.cancelReading()
    }
}

/// One frame from `VideoFrameReader`. The pixel buffer aliases AV
/// Foundation's internal buffer pool — finish using it (including any
/// GPU work) before pulling the next frame.
struct ReaderFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let presentationTime: CMTime
    let duration: CMTime
}
