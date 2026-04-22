//
//  RoughMatteKeyingEngine.swift
//  Corridor Key Pro
//
//  Fallback engine used when no neural model artefact is bundled (for example
//  during development builds or if the user disables network-download of
//  models). Produces a simple `max(G - max(R,B), 0)` matte that mirrors the
//  rough-matte helper in the CorridorKey CLI, so the plugin always has a
//  usable output even without ML.
//

import Foundation
import Metal

final class RoughMatteKeyingEngine: KeyingInferenceEngine, @unchecked Sendable {
    let backendDisplayName: String = "Rough Matte (Metal)"
    var guideSourceDescription: String = "Green-channel fallback"

    private let cacheEntry: MetalDeviceCacheEntry

    init(cacheEntry: MetalDeviceCacheEntry) {
        self.cacheEntry = cacheEntry
    }

    func supports(resolution: Int) -> Bool { true }

    func prepare(resolution: Int) async throws {
        // Nothing to do; all work happens per-frame via Metal.
    }

    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        // The render pipeline already committed and awaited the pre-inference
        // GPU pass, so we own the work here: grab a command buffer, dispatch
        // the rough-matte and resample kernels, then commit+wait so the
        // post-inference pass sees populated outputs.
        guard let commandQueue = cacheEntry.borrowCommandQueue() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        defer { cacheEntry.returnCommandQueue(commandQueue) }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        commandBuffer.label = "Corridor Key Rough Matte"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KeyingInferenceError.deviceUnavailable
        }

        // The normalised tensor texture already holds R=(source.r), G=(source.g),
        // B=(source.b), A=(hint). Reading it through the rough-matte kernel
        // produces a crude alpha from `max(G - max(R, B), 0)`.
        encoder.setComputePipelineState(cacheEntry.pipelines.roughMatte)
        encoder.setTexture(request.normalisedInputTexture, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.alphaTexture, index: Int(CKTextureIndexOutput.rawValue))
        let threadsAlpha = MTLSize(
            width: output.alphaTexture.width,
            height: output.alphaTexture.height,
            depth: 1
        )
        encoder.dispatchThreads(
            threadsAlpha,
            threadsPerThreadgroup: Self.threadgroupSize(for: cacheEntry.pipelines.roughMatte, threads: threadsAlpha)
        )

        // Foreground is the unmodified source at the inference resolution.
        // The despill pass downstream will remove any residual green cast.
        encoder.setComputePipelineState(cacheEntry.pipelines.resample)
        encoder.setTexture(request.normalisedInputTexture, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.foregroundTexture, index: Int(CKTextureIndexOutput.rawValue))
        let threadsFG = MTLSize(
            width: output.foregroundTexture.width,
            height: output.foregroundTexture.height,
            depth: 1
        )
        encoder.dispatchThreads(
            threadsFG,
            threadsPerThreadgroup: Self.threadgroupSize(for: cacheEntry.pipelines.resample, threads: threadsFG)
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    private static func threadgroupSize(
        for pipeline: MTLComputePipelineState,
        threads: MTLSize
    ) -> MTLSize {
        let width = min(pipeline.threadExecutionWidth, threads.width)
        let height = min(pipeline.maxTotalThreadsPerThreadgroup / max(width, 1), threads.height)
        return MTLSize(
            width: max(width, 1),
            height: max(height, 1),
            depth: 1
        )
    }
}
