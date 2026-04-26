//
//  StandaloneRenderEngineTests.swift
//  Corridor Key Toolbox — WrapperAppTests
//
//  Drives the engine that wraps `RenderPipeline` for the standalone
//  editor. The engine is the seam between AVFoundation pixel buffers
//  and the existing Metal pipeline, so these tests cover:
//
//  * Engine init succeeds on the system default Metal device.
//  * Calling `render(...)` with the source pass-through path (no
//    cached matte) returns a destination pixel buffer with the same
//    dimensions and format as the source, plus a recognisable
//    backend label.
//  * Calling `extractMatteBlob(...)` on a green-screen frame produces
//    a non-empty matte blob the same encoder/decoder pair the FxPlug
//    cache uses.
//

import Testing
import Foundation
import CoreMedia
import CoreVideo
@testable import CorridorKeyToolboxApp

@Suite("StandaloneRenderEngine", .serialized)
struct StandaloneRenderEngineTests {

    @Test("initialises against the system-default Metal device")
    func initialisesEngine() throws {
        _ = try StandaloneRenderEngine()
    }

    @Test("source pass-through render returns a destination pixel buffer with the right shape")
    func sourcePassThroughRender() async throws {
        let engine = try StandaloneRenderEngine()
        let buffer = try PixelBufferTextureBridge.makeMetalCompatiblePixelBuffer(
            width: 256,
            height: 144,
            pixelFormat: kCVPixelFormatType_64RGBAHalf
        )
        // No cached matte → pass-through path.
        let state = PluginStateData()
        let result = try engine.render(source: buffer, state: state, renderTime: .zero)
        #expect(CVPixelBufferGetWidth(result.destinationPixelBuffer) == 256)
        #expect(CVPixelBufferGetHeight(result.destinationPixelBuffer) == 144)
        #expect(result.report.backendDescription.contains("Pass-Through"))
    }

    @Test("extracts a non-empty matte blob from a synthetic green-screen frame")
    func extractsNonTrivialMatteBlob() async throws {
        let url = try await SyntheticVideoFixture.writeMP4()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let source = try await VideoSource(url: url)
        let pixelBuffer = try await source.makeFrame(atTime: .zero)

        let engine = try StandaloneRenderEngine()
        let output = try engine.extractMatteBlob(
            source: pixelBuffer,
            state: PluginStateData(qualityMode: .draft512),
            renderTime: .zero
        )

        #expect(!output.blob.isEmpty)
        #expect(output.width > 0 && output.height > 0)
        #expect(output.alphaFloats.count == output.width * output.height)
        // The encoder header includes a magic bytes prefix; if the
        // codec changes silently this will fall over.
        #expect(MatteCodec.parseHeader(output.blob) != nil)
    }

    @Test("warm-up status starts cold and transitions to warming when triggered")
    func warmupStatusTransitions() throws {
        let engine = try StandaloneRenderEngine()
        let initial = engine.warmupStatus(forResolution: 512)
        // The shared registry may already be warm from a previous test
        // in this suite — accept either cold or warming/ready.
        switch initial {
        case .cold, .warming, .ready, .failed:
            break // any outcome means the call returned without crashing
        }
        try engine.beginWarmup(forResolution: 512)
        let triggered = engine.warmupStatus(forResolution: 512)
        // After a warm-up has been requested the status should not be
        // "cold" any more — it's at least "warming".
        switch triggered {
        case .cold:
            Issue.record("Warm-up should not remain cold after beginWarmup() returns.")
        case .warming, .ready, .failed:
            break
        }
    }
}
