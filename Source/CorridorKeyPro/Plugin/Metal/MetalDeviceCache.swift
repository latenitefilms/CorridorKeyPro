//
//  MetalDeviceCache.swift
//  Corridor Key Pro
//
//  Caches Metal devices, command queues, pipeline states, and transient
//  textures keyed by the FxPlug `deviceRegistryID`. Final Cut Pro may hand the
//  plug-in tiles from multiple GPUs during the same session, and creating
//  those resources on the render path is measured in milliseconds — so we
//  create them lazily on first use and keep them alive for the XPC service's
//  lifetime.
//

import Foundation
import Metal

/// All of the compute pipelines we use during a render. Bundling them makes it
/// cheap to hand the full set to the `RenderPipeline` for a given device.
final class CorridorKeyPipelineSet: Sendable {
    let passthroughRender: MTLRenderPipelineState
    let applyColorMatrix: MTLRenderPipelineState

    let combineAndNormalize: MTLComputePipelineState
    let despill: MTLComputePipelineState
    let alphaLevelsGamma: MTLComputePipelineState
    let morphologyHorizontal: MTLComputePipelineState
    let morphologyVertical: MTLComputePipelineState
    let gaussianHorizontal: MTLComputePipelineState
    let gaussianVertical: MTLComputePipelineState
    let roughMatte: MTLComputePipelineState
    let sourcePassthrough: MTLComputePipelineState
    let compose: MTLComputePipelineState
    let resample: MTLComputePipelineState
    let extractHint: MTLComputePipelineState

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) throws {
        guard let library = device.makeDefaultLibrary() else {
            throw MetalDeviceCacheError.missingDefaultLibrary
        }

        func computePipeline(named name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalDeviceCacheError.missingShaderFunction(name)
            }
            return try device.makeComputePipelineState(function: function)
        }

        func renderPipeline(vertex: String, fragment: String, label: String) throws -> MTLRenderPipelineState {
            guard let vertexFunction = library.makeFunction(name: vertex) else {
                throw MetalDeviceCacheError.missingShaderFunction(vertex)
            }
            guard let fragmentFunction = library.makeFunction(name: fragment) else {
                throw MetalDeviceCacheError.missingShaderFunction(fragment)
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = label
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = fragmentFunction
            descriptor.colorAttachments[0].pixelFormat = pixelFormat
            return try device.makeRenderPipelineState(descriptor: descriptor)
        }

        passthroughRender = try renderPipeline(
            vertex: "corridorKeyVertexShader",
            fragment: "corridorKeyPassthroughFragment",
            label: "Corridor Key Passthrough"
        )
        applyColorMatrix = try renderPipeline(
            vertex: "corridorKeyVertexShader",
            fragment: "corridorKeyApplyColorMatrixFragment",
            label: "Corridor Key Colour Matrix"
        )
        combineAndNormalize = try computePipeline(named: "corridorKeyCombineAndNormalizeKernel")
        despill = try computePipeline(named: "corridorKeyDespillKernel")
        alphaLevelsGamma = try computePipeline(named: "corridorKeyAlphaLevelsGammaKernel")
        morphologyHorizontal = try computePipeline(named: "corridorKeyMorphologyHorizontalKernel")
        morphologyVertical = try computePipeline(named: "corridorKeyMorphologyVerticalKernel")
        gaussianHorizontal = try computePipeline(named: "corridorKeyGaussianHorizontalKernel")
        gaussianVertical = try computePipeline(named: "corridorKeyGaussianVerticalKernel")
        roughMatte = try computePipeline(named: "corridorKeyRoughMatteKernel")
        sourcePassthrough = try computePipeline(named: "corridorKeySourcePassthroughKernel")
        compose = try computePipeline(named: "corridorKeyComposeKernel")
        resample = try computePipeline(named: "corridorKeyResampleKernel")
        extractHint = try computePipeline(named: "corridorKeyExtractHintKernel")
    }
}

enum MetalDeviceCacheError: Error, CustomStringConvertible {
    case missingDefaultLibrary
    case missingShaderFunction(String)
    case unknownDevice(UInt64)
    case queueExhausted

    var description: String {
        switch self {
        case .missingDefaultLibrary:
            return "Corridor Key Pro could not locate its compiled Metal library."
        case .missingShaderFunction(let name):
            return "Corridor Key Pro could not find Metal function \(name)."
        case .unknownDevice(let registryID):
            return "Corridor Key Pro was handed an unfamiliar GPU (registry id \(registryID))."
        case .queueExhausted:
            return "All Corridor Key Pro command queues are currently in flight."
        }
    }
}

/// Single entry in the cache, one per (device, pixel format) pair.
final class MetalDeviceCacheEntry {
    let device: MTLDevice
    let pixelFormat: MTLPixelFormat
    let pipelines: CorridorKeyPipelineSet

    private let queueLock = NSLock()
    private var commandQueues: [MTLCommandQueue]
    private var availability: [Bool]

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) throws {
        self.device = device
        self.pixelFormat = pixelFormat
        self.pipelines = try CorridorKeyPipelineSet(device: device, pixelFormat: pixelFormat)

        let queueCount = 5
        var queues: [MTLCommandQueue] = []
        queues.reserveCapacity(queueCount)
        for _ in 0..<queueCount {
            if let queue = device.makeCommandQueue() {
                queue.label = "Corridor Key Pro Queue"
                queues.append(queue)
            }
        }
        self.commandQueues = queues
        self.availability = Array(repeating: true, count: queues.count)
    }

    /// Checks out a command queue for the duration of a render. Callers must
    /// return the queue via `returnCommandQueue(_:)` to avoid starving the pool.
    func borrowCommandQueue() -> MTLCommandQueue? {
        queueLock.lock()
        defer { queueLock.unlock() }
        for index in availability.indices where availability[index] {
            availability[index] = false
            return commandQueues[index]
        }
        return nil
    }

    func returnCommandQueue(_ queue: MTLCommandQueue) {
        queueLock.lock()
        defer { queueLock.unlock() }
        for index in commandQueues.indices where commandQueues[index] === queue {
            availability[index] = true
            return
        }
    }

    /// Creates a throwaway texture matching the supplied size. Intermediate
    /// GPU-only textures default to `.private` storage for the best GPU
    /// bandwidth. Textures that need to be read or written by the CPU – for
    /// example the buffers we hand to MLX – must be allocated with
    /// `.shared` storage so Metal allows `getBytes` / `replace(region:…)`.
    func makeIntermediateTexture(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .rgba16Float,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite],
        storageMode: MTLStorageMode = .private
    ) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: max(width, 1),
            height: max(height, 1),
            mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = storageMode
        return device.makeTexture(descriptor: descriptor)
    }
}

/// Singleton cache shared across every `CorridorKeyProPlugIn` instance living in
/// the XPC service. Each instance still owns its own `Engine` (see the inference
/// layer) but they all share the compiled Metal artefacts, which is the
/// expensive part.
final class MetalDeviceCache: @unchecked Sendable {
    static let shared = MetalDeviceCache()

    private let entriesLock = NSLock()
    private var entries: [MetalDeviceCacheEntry] = []

    /// Returns the cache entry for a given GPU + pixel format combo, creating
    /// it lazily on first use.
    func entry(for device: MTLDevice, pixelFormat: MTLPixelFormat) throws -> MetalDeviceCacheEntry {
        entriesLock.lock()
        defer { entriesLock.unlock() }

        for entry in entries where entry.device.registryID == device.registryID && entry.pixelFormat == pixelFormat {
            return entry
        }
        let newEntry = try MetalDeviceCacheEntry(device: device, pixelFormat: pixelFormat)
        entries.append(newEntry)
        return newEntry
    }

    /// Looks up a device by the registry identifier FxPlug provides on each
    /// `FxImageTile`.
    func device(forRegistryID registryID: UInt64) -> MTLDevice? {
        for device in MTLCopyAllDevices() where device.registryID == registryID {
            return device
        }
        return nil
    }

    /// Translates the IOSurface pixel format FxPlug gives us into the nearest
    /// Metal pixel format. Falls back to `rgba16Float` because Final Cut Pro
    /// prefers half-float tiles whenever the host is colour-managed.
    static func metalPixelFormat(for tile: FxImageTile) -> MTLPixelFormat {
        switch tile.ioSurface.pixelFormat {
        case kCVPixelFormatType_128RGBAFloat: return .rgba32Float
        case kCVPixelFormatType_64RGBAHalf: return .rgba16Float
        case kCVPixelFormatType_32BGRA: return .bgra8Unorm
        default: return .rgba16Float
        }
    }
}
