//
//  SyntheticVideoFixture.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  Builds a tiny synthetic MP4 on disk for the wrapper-app tests so we
//  don't have to bundle a 130 MB sample clip into the test bundle. The
//  generated clip is 320 × 180 RGBA, 24 fps, 12 frames (half a second),
//  with a green-bias gradient in the centre — close enough to a real
//  green-screen frame that the Corridor Key pipeline produces a
//  non-trivial matte on it.
//
//  Each frame is a deterministic gradient so tests can compare
//  consecutive renders without flakiness.
//

import Foundation
import AVFoundation
import CoreVideo
import CoreImage
import CoreImage.CIFilterBuiltins

enum SyntheticVideoFixture {

    /// Generates an `Input.mp4` inside a temporary directory and
    /// returns the URL. Caller is responsible for deleting the
    /// directory tree if they want a clean slate; XCTest's
    /// `temporaryDirectory` for the test run handles this for us.
    static func writeMP4(
        width: Int = 320,
        height: Int = 180,
        frameCount: Int = 12,
        fps: Int = 24
    ) async throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "CorridorKeyTests", directoryHint: .isDirectory)
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "Input.mp4")

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264.rawValue,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = false
        let pixelBufferAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: pixelBufferAttrs
        )
        guard writer.canAdd(input) else {
            throw FixtureError.writerRejectedInput
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw writer.error ?? FixtureError.writerStartFailed
        }
        writer.startSession(atSourceTime: .zero)

        let timeScale = CMTimeScale(fps * 1000)
        let frameDurationTicks = Int64(timeScale / Int32(fps))
        let context = CIContext(options: [.useSoftwareRenderer: false])

        for index in 0..<frameCount {
            // Block until the writer is ready before pushing the next
            // frame; the in-memory buffer is bounded.
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(2))
            }
            let pixelBuffer = try makeBuffer(
                width: width,
                height: height,
                frameIndex: index,
                frameCount: frameCount,
                context: context
            )
            let presentationTime = CMTime(
                value: Int64(index) * frameDurationTicks,
                timescale: timeScale
            )
            if !adaptor.append(pixelBuffer, withPresentationTime: presentationTime) {
                throw writer.error ?? FixtureError.appendFailed(index: index)
            }
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw writer.error ?? FixtureError.writerDidNotComplete(status: writer.status.rawValue)
        }
        return url
    }

    /// Builds one synthetic frame: a green background with a moving
    /// foreground bar so the temporal-stability path has something to
    /// gate motion on.
    private static func makeBuffer(
        width: Int,
        height: Int,
        frameIndex: Int,
        frameCount: Int,
        context: CIContext
    ) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw FixtureError.allocationFailed(status: status)
        }
        let extent = CGRect(x: 0, y: 0, width: width, height: height)
        // Green background — 0.08, 0.84, 0.08 matches the canonical
        // green-screen colour the renderer's despill is tuned for.
        let background = CIImage(color: CIColor(red: 0.08, green: 0.84, blue: 0.08))
            .cropped(to: extent)
        // Foreground "subject": a small white rectangle that walks
        // horizontally across the frame.
        let subjectWidth = CGFloat(width) * 0.20
        let subjectHeight = CGFloat(height) * 0.50
        let progress = CGFloat(frameIndex) / CGFloat(max(frameCount - 1, 1))
        let originX = (CGFloat(width) - subjectWidth) * progress
        let originY = (CGFloat(height) - subjectHeight) * 0.5
        let subjectRect = CGRect(x: originX, y: originY, width: subjectWidth, height: subjectHeight)
        let subject = CIImage(color: CIColor(red: 0.95, green: 0.85, blue: 0.7))
            .cropped(to: subjectRect)
        let composed = subject.composited(over: background)
        context.render(
            composed,
            to: buffer,
            bounds: extent,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )
        return buffer
    }

    enum FixtureError: Error, CustomStringConvertible {
        case writerRejectedInput
        case writerStartFailed
        case appendFailed(index: Int)
        case writerDidNotComplete(status: Int)
        case allocationFailed(status: CVReturn)

        var description: String {
            switch self {
            case .writerRejectedInput: return "AVAssetWriter rejected the synthetic video input."
            case .writerStartFailed: return "AVAssetWriter could not start."
            case .appendFailed(let index): return "Failed to append frame \(index) to the synthetic clip."
            case .writerDidNotComplete(let status): return "AVAssetWriter finished with status \(status)."
            case .allocationFailed(let status): return "CVPixelBufferCreate returned \(status)."
            }
        }
    }
}
