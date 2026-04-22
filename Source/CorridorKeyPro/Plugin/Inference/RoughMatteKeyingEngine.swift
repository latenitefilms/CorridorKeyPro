//
//  RoughMatteKeyingEngine.swift
//  Corridor Key Pro
//
//  Default engine used whenever MLX is disabled or unavailable. Computes the
//  matte from the raw (0..1) RGB source using the `corridorKeyRoughMatteKernel`
//  — a simple `1 - saturate((G - max(R, B)) * 2.5)` that cleanly matches the
//  convention used by CorridorKey-Runtime's `ColorUtils::generate_rough_matte`.
//

import Foundation
import Metal

final class RoughMatteKeyingEngine: KeyingInferenceEngine, @unchecked Sendable {
    let backendDisplayName: String
    var guideSourceDescription: String = "Green-channel fallback"

    private let cacheEntry: MetalDeviceCacheEntry

    init(cacheEntry: MetalDeviceCacheEntry) {
        self.cacheEntry = cacheEntry
        self.backendDisplayName = "Rough Matte (Metal) on \(cacheEntry.device.name)"
    }

    func supports(resolution: Int) -> Bool { true }

    func prepare(resolution: Int) async throws {
        // Nothing to prepare; all work happens per-frame on the GPU.
    }

    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        guard let commandQueue = cacheEntry.borrowCommandQueue() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        defer { cacheEntry.returnCommandQueue(commandQueue) }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        commandBuffer.label = "Corridor Key Pro Rough Matte"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KeyingInferenceError.deviceUnavailable
        }

        // Derive the matte from the raw RGB source (not the ImageNet-normalised
        // tensor) so the green-detection math operates in 0..1 space and the
        // fallback matches the reference CPU rough-matte exactly.
        encoder.setComputePipelineState(cacheEntry.computePipelines.roughMatte)
        encoder.setTexture(request.rawSourceTexture, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.alphaTexture, index: Int(CKTextureIndexOutput.rawValue))
        dispatchThreads(
            encoder: encoder,
            pipeline: cacheEntry.computePipelines.roughMatte,
            width: output.alphaTexture.width,
            height: output.alphaTexture.height
        )

        // The foreground texture is only used by downstream despill /
        // passthrough stages that read the ML-provided foreground tensor.
        // Fill it with the raw RGB source so those stages see sensible
        // values even when MLX is disabled.
        encoder.setComputePipelineState(cacheEntry.computePipelines.resample)
        encoder.setTexture(request.rawSourceTexture, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.foregroundTexture, index: Int(CKTextureIndexOutput.rawValue))
        dispatchThreads(
            encoder: encoder,
            pipeline: cacheEntry.computePipelines.resample,
            width: output.foregroundTexture.width,
            height: output.foregroundTexture.height
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
    }

    private func dispatchThreads(
        encoder: any MTLComputeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        let threadgroupWidth = min(pipeline.threadExecutionWidth, max(width, 1))
        let threadgroupHeight = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / max(threadgroupWidth, 1), max(height, 1)))
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        let threadsPerGrid = MTLSize(width: max(width, 1), height: max(height, 1), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }
}
