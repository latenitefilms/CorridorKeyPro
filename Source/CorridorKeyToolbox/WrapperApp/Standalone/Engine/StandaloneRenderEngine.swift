//
//  StandaloneRenderEngine.swift
//  Corridor Key Toolbox — Standalone Editor
//
//  Glue layer between the standalone editor and the shared render
//  pipeline. Holds onto the system-default `MTLDevice`, a single
//  `RenderPipeline` instance, and a `PixelBufferTextureBridge` so each
//  preview / export call boils down to:
//
//      1. Wrap the source CVPixelBuffer as an MTLTexture (zero copy).
//      2. Make a Metal-compatible destination CVPixelBuffer if we're
//         exporting, or reuse the preview output texture.
//      3. Call `RenderPipeline.renderToTexture(...)`.
//      4. Hand the destination CVPixelBuffer back to the caller so it
//         can either upload it to a CAMetalLayer (preview) or write it
//         to AVAssetWriter (export).
//
//  The same render pipeline used by the FxPlug also powers this engine,
//  guaranteeing visual parity between Final Cut Pro and the standalone
//  editor without code duplication.
//

import Foundation
import AVFoundation
import CoreMedia
import CoreVideo
import Metal

/// Errors surfaced to the standalone editor when a render fails.
enum StandaloneRenderEngineError: Error, CustomStringConvertible {
    case noMetalDevice
    case bridgeUnavailable
    case destinationAllocationFailed

    var description: String {
        switch self {
        case .noMetalDevice:
            return "Corridor Key Toolbox could not access the system Metal device."
        case .bridgeUnavailable:
            return "Corridor Key Toolbox could not initialise the pixel-buffer ↔ Metal bridge."
        case .destinationAllocationFailed:
            return "Corridor Key Toolbox could not allocate a destination texture."
        }
    }
}

/// Bundle returned from a render. Holds the destination pixel buffer
/// (for downstream encoders), the Metal texture aliased onto its
/// IOSurface (for preview display), and the renderer's status report
/// so the editor can show "Source Pass-Through" / "Cached MLX" badges.
struct StandaloneRenderResult: @unchecked Sendable {
    let destinationPixelBuffer: CVPixelBuffer
    let destinationTexture: any MTLTexture
    let report: RenderReport
}

/// One-stop render facade for the standalone editor. Constructed once
/// per editor session; held by `EditorViewModel` and shared across
/// preview, analyse, and export.
final class StandaloneRenderEngine: @unchecked Sendable {

    let device: any MTLDevice
    private let pipeline: RenderPipeline
    private let bridge: PixelBufferTextureBridge

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw StandaloneRenderEngineError.noMetalDevice
        }
        self.device = device
        self.pipeline = RenderPipeline()
        do {
            self.bridge = try PixelBufferTextureBridge(device: device)
        } catch {
            throw StandaloneRenderEngineError.bridgeUnavailable
        }
    }

    // MARK: - Preview / scrub

    /// Renders one frame for preview. The destination uses the same
    /// pixel format as the source so the preview layer can display it
    /// directly without an extra blit. The destination texture is
    /// created with `[.shaderRead, .renderTarget]` usage so the
    /// pipeline's compose render pass can bind it as a colour
    /// attachment.
    func render(
        source pixelBuffer: CVPixelBuffer,
        state: PluginStateData,
        renderTime: CMTime
    ) throws -> StandaloneRenderResult {
        let sourceBacked = try bridge.makeTexture(for: pixelBuffer, usage: .shaderRead)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let destinationBuffer = try PixelBufferTextureBridge.makeMetalCompatiblePixelBuffer(
            width: sourceBacked.metalTexture.width,
            height: sourceBacked.metalTexture.height,
            pixelFormat: pixelFormat
        )
        let destinationBacked = try bridge.makeTexture(
            for: destinationBuffer,
            usage: [.shaderRead, .renderTarget]
        )
        let report = try pipeline.renderToTexture(
            source: sourceBacked.metalTexture,
            destination: destinationBacked.metalTexture,
            alphaHint: nil,
            state: state,
            workingGamut: .rec709,
            renderTime: renderTime
        )
        return StandaloneRenderResult(
            destinationPixelBuffer: destinationBuffer,
            destinationTexture: destinationBacked.metalTexture,
            report: report
        )
    }

    // MARK: - Analysis

    /// Runs the FxAnalyzer-equivalent extraction step on a single
    /// frame. Returns the encoded matte blob ready to be stored in the
    /// project's matte cache. Mirrors `RenderPipeline.extractAlphaMatteForAnalysis`
    /// but exposes only the bits the standalone editor needs.
    func extractMatteBlob(
        source pixelBuffer: CVPixelBuffer,
        state: PluginStateData,
        renderTime: CMTime
    ) throws -> AnalysisFrameOutput {
        let sourceBacked = try bridge.makeTexture(for: pixelBuffer, usage: .shaderRead)
        let device = sourceBacked.metalTexture.device
        let entry = try MetalDeviceCache.shared.entry(for: device)
        guard let commandQueue = entry.borrowCommandQueue() else {
            throw MetalDeviceCacheError.queueExhausted
        }
        defer { entry.returnCommandQueue(commandQueue) }
        let extracted = try pipeline.extractAlphaMatteForAnalysis(
            sourceTexture: sourceBacked.metalTexture,
            state: state,
            workingGamut: .rec709,
            renderTime: renderTime,
            device: device,
            entry: entry,
            commandQueue: commandQueue,
            readbackSource: state.temporalStabilityEnabled
        )
        let blob = try MatteCodec.encode(
            alpha: extracted.alpha,
            width: extracted.width,
            height: extracted.height
        )
        return AnalysisFrameOutput(
            blob: blob,
            width: extracted.width,
            height: extracted.height,
            inferenceResolution: extracted.inferenceResolution,
            engineDescription: extracted.engineDescription,
            sourceFloats: extracted.source,
            alphaFloats: extracted.alpha
        )
    }

    /// Eagerly warms up the MLX bridge for the chosen quality rung.
    /// Lets the editor surface a "Loading neural model…" status while
    /// the bridge compiles, the same way the FxPlug does.
    func beginWarmup(forResolution resolution: Int) throws {
        let entry = try MetalDeviceCache.shared.entry(for: device)
        SharedMLXBridgeRegistry.shared.beginWarmup(
            deviceRegistryID: device.registryID,
            rung: resolution,
            cacheEntry: entry
        )
    }

    /// Queries the warm-up status for the inspector status badge.
    func warmupStatus(forResolution resolution: Int) -> WarmupStatus {
        SharedMLXBridgeRegistry.shared.status(
            deviceRegistryID: device.registryID,
            rung: resolution
        )
    }
}

/// Per-frame analysis output. Kept separate from the render result so
/// callers cannot accidentally route it back into the preview path.
struct AnalysisFrameOutput: @unchecked Sendable {
    let blob: Data
    let width: Int
    let height: Int
    let inferenceResolution: Int
    let engineDescription: String
    /// Optional raw float buffer of the source RGBA at inference
    /// resolution. Populated only when temporal stability is on so the
    /// CPU-side blender can compute motion gates.
    let sourceFloats: [Float]?
    /// Raw alpha matte at inference resolution. The same data the blob
    /// encodes; included separately so the temporal blender can mutate
    /// it without re-decoding.
    let alphaFloats: [Float]
}
