//
//  MetalDeviceCache.swift
//  Corridor Key Pro
//
//  Caches Metal devices, command queues, and compiled pipeline states so the
//  render path never does I/O. Compute pipelines are cached per device; render
//  pipelines (which depend on the destination pixel format) are cached per
//  (device, pixel format) pair. Final Cut Pro may hand us tiles from multiple
//  GPUs within the same session.
//

import Foundation
import Metal

enum MetalDeviceCacheError: Error, CustomStringConvertible {
    case missingDefaultLibrary
    case missingShaderFunction(String)
    case unknownDevice(UInt64)
    case queueExhausted
    case textureAllocationFailed
    case commandBufferCreationFailed
    case commandEncoderCreationFailed

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
        case .textureAllocationFailed:
            return "Corridor Key Pro could not allocate an intermediate Metal texture."
        case .commandBufferCreationFailed:
            return "Corridor Key Pro could not create a Metal command buffer."
        case .commandEncoderCreationFailed:
            return "Corridor Key Pro could not create a Metal command encoder."
        }
    }
}

/// Compute pipelines keyed only by device. They do not depend on the output
/// pixel format because compute shaders write untyped floats.
final class CorridorKeyComputePipelines: Sendable {
    let combineAndNormalize: any MTLComputePipelineState
    let despill: any MTLComputePipelineState
    let alphaLevelsGamma: any MTLComputePipelineState
    let morphologyHorizontal: any MTLComputePipelineState
    let morphologyVertical: any MTLComputePipelineState
    let gaussianHorizontal: any MTLComputePipelineState
    let gaussianVertical: any MTLComputePipelineState
    let roughMatte: any MTLComputePipelineState
    let greenHint: any MTLComputePipelineState
    let sourcePassthrough: any MTLComputePipelineState
    let applyScreenMatrix: any MTLComputePipelineState
    let resample: any MTLComputePipelineState
    let extractHint: any MTLComputePipelineState

    init(device: any MTLDevice, library: any MTLLibrary) throws {
        func compute(_ name: String) throws -> any MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalDeviceCacheError.missingShaderFunction(name)
            }
            return try device.makeComputePipelineState(function: function)
        }
        combineAndNormalize = try compute("corridorKeyCombineAndNormalizeKernel")
        despill = try compute("corridorKeyDespillKernel")
        alphaLevelsGamma = try compute("corridorKeyAlphaLevelsGammaKernel")
        morphologyHorizontal = try compute("corridorKeyMorphologyHorizontalKernel")
        morphologyVertical = try compute("corridorKeyMorphologyVerticalKernel")
        gaussianHorizontal = try compute("corridorKeyGaussianHorizontalKernel")
        gaussianVertical = try compute("corridorKeyGaussianVerticalKernel")
        roughMatte = try compute("corridorKeyRoughMatteKernel")
        greenHint = try compute("corridorKeyGreenHintKernel")
        sourcePassthrough = try compute("corridorKeySourcePassthroughKernel")
        applyScreenMatrix = try compute("corridorKeyApplyScreenMatrixKernel")
        resample = try compute("corridorKeyResampleKernel")
        extractHint = try compute("corridorKeyExtractHintKernel")
    }
}

/// Render pipelines depend on the destination pixel format (the colour
/// attachment format). The compose pipeline reads the per-frame intermediates
/// and writes the final RGBA to Final Cut Pro's destination texture.
final class CorridorKeyRenderPipelines: Sendable {
    let compose: any MTLRenderPipelineState

    init(device: any MTLDevice, library: any MTLLibrary, pixelFormat: MTLPixelFormat) throws {
        guard let vertexFunction = library.makeFunction(name: "corridorKeyComposeVertex") else {
            throw MetalDeviceCacheError.missingShaderFunction("corridorKeyComposeVertex")
        }
        guard let fragmentFunction = library.makeFunction(name: "corridorKeyComposeFragment") else {
            throw MetalDeviceCacheError.missingShaderFunction("corridorKeyComposeFragment")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Corridor Key Pro Compose"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        compose = try device.makeRenderPipelineState(descriptor: descriptor)
    }
}

/// Per-device state: the shared Metal library, compiled compute pipelines,
/// a pool of command queues, and a per-format cache of render pipelines.
final class MetalDeviceCacheEntry {
    let device: any MTLDevice
    let library: any MTLLibrary
    let computePipelines: CorridorKeyComputePipelines

    private let queueLock = NSLock()
    private var commandQueues: [any MTLCommandQueue]
    private var availability: [Bool]

    private let renderPipelinesLock = NSLock()
    private var renderPipelines: [MTLPixelFormat: CorridorKeyRenderPipelines] = [:]

    init(device: any MTLDevice) throws {
        self.device = device

        guard let library = device.makeDefaultLibrary() else {
            throw MetalDeviceCacheError.missingDefaultLibrary
        }
        self.library = library
        self.computePipelines = try CorridorKeyComputePipelines(device: device, library: library)

        let queueCount = 4
        var queues: [any MTLCommandQueue] = []
        queues.reserveCapacity(queueCount)
        for _ in 0..<queueCount {
            if let queue = device.makeCommandQueue() {
                queue.label = "Corridor Key Pro"
                queues.append(queue)
            }
        }
        self.commandQueues = queues
        self.availability = Array(repeating: true, count: queues.count)
    }

    func borrowCommandQueue() -> (any MTLCommandQueue)? {
        queueLock.lock()
        defer { queueLock.unlock() }
        for index in availability.indices where availability[index] {
            availability[index] = false
            return commandQueues[index]
        }
        return nil
    }

    func returnCommandQueue(_ queue: any MTLCommandQueue) {
        queueLock.lock()
        defer { queueLock.unlock() }
        for index in commandQueues.indices where commandQueues[index] === queue {
            availability[index] = true
            return
        }
    }

    func renderPipelines(for pixelFormat: MTLPixelFormat) throws -> CorridorKeyRenderPipelines {
        renderPipelinesLock.lock()
        if let existing = renderPipelines[pixelFormat] {
            renderPipelinesLock.unlock()
            return existing
        }
        renderPipelinesLock.unlock()

        let created = try CorridorKeyRenderPipelines(device: device, library: library, pixelFormat: pixelFormat)
        renderPipelinesLock.lock()
        renderPipelines[pixelFormat] = created
        renderPipelinesLock.unlock()
        return created
    }

    /// Allocates an intermediate texture used between passes. Defaults to
    /// `.private` storage because most intermediates never cross the CPU.
    /// Inference engines require `.shared` so their input/output buffers can
    /// be read and written from the CPU.
    func makeIntermediateTexture(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .rgba16Float,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite],
        storageMode: MTLStorageMode = .private
    ) -> (any MTLTexture)? {
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

/// Singleton cache shared by every plug-in instance in the XPC service.
final class MetalDeviceCache: @unchecked Sendable {
    static let shared = MetalDeviceCache()

    private let entriesLock = NSLock()
    private var entries: [UInt64: MetalDeviceCacheEntry] = [:]

    func entry(for device: any MTLDevice) throws -> MetalDeviceCacheEntry {
        entriesLock.lock()
        defer { entriesLock.unlock() }
        if let existing = entries[device.registryID] {
            return existing
        }
        let newEntry = try MetalDeviceCacheEntry(device: device)
        entries[device.registryID] = newEntry
        return newEntry
    }

    func device(forRegistryID registryID: UInt64) -> (any MTLDevice)? {
        for device in MTLCopyAllDevices() where device.registryID == registryID {
            return device
        }
        return nil
    }

    /// Maps the IOSurface pixel format FxPlug provides to the nearest Metal
    /// pixel format. Modern Final Cut Pro hands us 16-bit half floats for
    /// colour-managed output and BGRA8 for quick draft renders.
    static func metalPixelFormat(for tile: FxImageTile) -> MTLPixelFormat {
        switch tile.ioSurface.pixelFormat {
        case kCVPixelFormatType_128RGBAFloat: return .rgba32Float
        case kCVPixelFormatType_64RGBAHalf: return .rgba16Float
        case kCVPixelFormatType_32BGRA: return .bgra8Unorm
        default:
            PluginLog.warning(
                "Unexpected IOSurface pixel format \(String(tile.ioSurface.pixelFormat, radix: 16)); defaulting to rgba16Float."
            )
            return .rgba16Float
        }
    }
}
