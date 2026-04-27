//
//  VideoSourceTests.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  Verifies the AVFoundation import path the standalone editor uses
//  to bring a clip into the keyer. These are the tests that would
//  have caught the original "permission denied" regression: opening
//  a synthetic MP4 in `/var/folders/...` requires AV Foundation to
//  actually read the file, and the bridge has to handle the URL the
//  way the production import flow does.
//

import Testing
import Foundation
import AVFoundation
import CoreMedia
@testable import CorridorKeyByLateNiteApp

@Suite("VideoSource", .serialized)
struct VideoSourceTests {

    @Test("opens a synthetic MP4 and reports its dimensions, frame rate, and duration")
    func opensSyntheticMP4() async throws {
        let url = try await SyntheticVideoFixture.writeMP4()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let source = try await VideoSource(url: url)
        #expect(source.info.renderSize == CGSize(width: 320, height: 180))
        #expect(source.info.nominalFrameRate.rounded() == 24)
        #expect(source.info.duration.seconds >= 0.45 && source.info.duration.seconds <= 0.55)
    }

    @Test("reports the same frame count as the writer wrote")
    func totalFrameCountMatchesFixture() async throws {
        let url = try await SyntheticVideoFixture.writeMP4(frameCount: 12, fps: 24)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let source = try await VideoSource(url: url)
        #expect(source.totalFrameCount() == 12)
    }

    @Test("scrubs to the nearest frame boundary instead of returning sub-frame times")
    func scrubsToNearestFrame() async throws {
        let url = try await SyntheticVideoFixture.writeMP4(fps: 24)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let source = try await VideoSource(url: url)
        let request = CMTime(seconds: 0.20, preferredTimescale: 24_000)
        let snapped = source.nearestFrameTime(to: request)
        let frameRate = Double(source.info.nominalFrameRate)
        let snappedFrame = (snapped.seconds * frameRate).rounded()
        #expect(abs(snapped.seconds - snappedFrame / frameRate) < 1.0e-6)
    }

    @Test("decodes a single frame via the image generator without permission errors")
    func makeFrameDecodesFirstFrame() async throws {
        let url = try await SyntheticVideoFixture.writeMP4()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let source = try await VideoSource(url: url)
        let pixelBuffer = try await source.makeFrame(atTime: .zero)
        #expect(CVPixelBufferGetWidth(pixelBuffer) == 320)
        #expect(CVPixelBufferGetHeight(pixelBuffer) == 180)
    }

    @Test("frame reader walks every frame in display order")
    func frameReaderEnumeratesAllFrames() async throws {
        let url = try await SyntheticVideoFixture.writeMP4(frameCount: 12, fps: 24)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let source = try await VideoSource(url: url)
        let reader = try source.makeFrameReader()
        defer { reader.cancel() }

        var seen = 0
        var lastPresentationTime = CMTime.invalid
        while let frame = try reader.nextFrame() {
            #expect(CVPixelBufferGetWidth(frame.pixelBuffer) == 320)
            if lastPresentationTime.isNumeric {
                #expect(frame.presentationTime > lastPresentationTime)
            }
            lastPresentationTime = frame.presentationTime
            seen += 1
            if seen > 100 { break } // Safety net.
        }
        #expect(seen == 12)
    }

    @Test("throws a descriptive error when the URL is missing")
    func throwsForMissingFile() async throws {
        let bogus = FileManager.default.temporaryDirectory
            .appending(path: "Definitely-Not-A-Real-File-\(UUID().uuidString).mp4")
        await #expect(throws: (any Error).self) {
            _ = try await VideoSource(url: bogus)
        }
    }
}
