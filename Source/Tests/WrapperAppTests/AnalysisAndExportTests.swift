//
//  AnalysisAndExportTests.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  End-to-end tests for the standalone editor's two background
//  passes:
//
//  * `AnalysisRunner` — walks every frame of a clip, builds matte
//    blobs, fires status events.
//  * `ProResExporter` — replays the clip through the keyer using the
//    cached mattes and writes a ProRes 4444 .mov.
//
//  Both paths use a synthetic 12-frame clip so the suite finishes
//  in a few seconds even on the slowest M-series chip we ship to.
//

import Testing
import Foundation
import AVFoundation
import CoreMedia
@testable import CorridorKeyToolboxApp

@Suite("AnalysisRunner & ProResExporter", .serialized)
struct AnalysisAndExportTests {

    @Test("analyse pass walks every frame and reports completion")
    func analysisCoversEveryFrame() async throws {
        let url = try await SyntheticVideoFixture.writeMP4(frameCount: 8, fps: 24)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let engine = try StandaloneRenderEngine()
        let source = try await VideoSource(url: url)
        let runner = AnalysisRunner(renderEngine: engine, videoSource: source)

        let collector = EventCollector<AnalysisRunnerEvent>()
        await runner.run(state: PluginStateData(qualityMode: .draft512)) { event in
            collector.append(event)
        }
        let events = collector.snapshot()

        let processedCount = events.reduce(into: 0) { count, event in
            if case .frameProcessed = event { count += 1 }
        }
        #expect(processedCount == 8)
        let completed = events.contains { event in
            if case .completed = event { return true }
            return false
        }
        #expect(completed)
    }

    @Test("matte cache survives the analyse pass with one entry per source frame")
    func analysisFillsMatteCache() async throws {
        let url = try await SyntheticVideoFixture.writeMP4(frameCount: 6, fps: 24)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let engine = try StandaloneRenderEngine()
        let source = try await VideoSource(url: url)
        let runner = AnalysisRunner(renderEngine: engine, videoSource: source)

        let cacheCollector = MatteCacheCollector()
        await runner.run(state: PluginStateData(qualityMode: .draft512)) { event in
            if case let .frameProcessed(index, _, entry) = event {
                cacheCollector.set(index: index, entry: entry)
            }
        }
        let cache = cacheCollector.snapshot()
        #expect(cache.count == 6)
        for entry in cache.values {
            #expect(!entry.blob.isEmpty)
            #expect(entry.inferenceResolution > 0)
        }
    }

    @Test("export writes a playable ProRes 4444 .mov of the same length")
    func exportProducesPlayableMOV() async throws {
        let inputURL = try await SyntheticVideoFixture.writeMP4(frameCount: 6, fps: 24)
        defer { try? FileManager.default.removeItem(at: inputURL.deletingLastPathComponent()) }

        let engine = try StandaloneRenderEngine()
        let source = try await VideoSource(url: inputURL)

        // Run analysis so the exporter has cached mattes to feed.
        let runner = AnalysisRunner(renderEngine: engine, videoSource: source)
        let cacheCollector = MatteCacheCollector()
        await runner.run(state: PluginStateData(qualityMode: .draft512)) { event in
            if case let .frameProcessed(index, _, entry) = event {
                cacheCollector.set(index: index, entry: entry)
            }
        }
        let cache = cacheCollector.snapshot()

        let outputURL = inputURL.deletingLastPathComponent()
            .appending(path: "Output.mov")
        let snapshot = ExportProjectSnapshot(
            state: PluginStateData(qualityMode: .draft512),
            cachedMattes: cache
        )
        let exporter = ProResExporter(
            renderEngine: engine,
            videoSource: source,
            project: snapshot
        )
        let collector = EventCollector<ExportRunnerEvent>()
        await exporter.run(options: ExportOptions(
            destination: outputURL,
            codec: .proRes4444,
            preserveAlpha: true
        )) { event in
            collector.append(event)
        }
        let events = collector.snapshot()

        let completed = events.contains { event in
            if case .completed = event { return true }
            return false
        }
        #expect(completed, "Export did not complete: \(events)")
        #expect(FileManager.default.fileExists(atPath: outputURL.path))

        // Verify the written file is a real movie AV Foundation can
        // read back — same gate FCP would apply on import.
        let writtenAsset = AVURLAsset(url: outputURL)
        let writtenTracks = try await writtenAsset.loadTracks(withMediaType: .video)
        #expect(writtenTracks.count == 1)
        let writtenDuration = try await writtenAsset.load(.duration)
        #expect(writtenDuration.seconds > 0.20)
    }
}

/// Tiny thread-safe collector for `@Sendable` event closures.
final class EventCollector<Event>: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [Event] = []

    func append(_ event: Event) {
        lock.lock(); defer { lock.unlock() }
        events.append(event)
    }

    func snapshot() -> [Event] {
        lock.lock(); defer { lock.unlock() }
        return events
    }
}

/// Thread-safe matte-cache assembler so a `@Sendable` event handler
/// can drop per-frame entries into a shared dictionary without falling
/// foul of Swift 6 strict concurrency.
final class MatteCacheCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [Int: MatteCacheEntry] = [:]

    func set(index: Int, entry: MatteCacheEntry) {
        lock.lock(); defer { lock.unlock() }
        entries[index] = entry
    }

    func snapshot() -> [Int: MatteCacheEntry] {
        lock.lock(); defer { lock.unlock() }
        return entries
    }
}
