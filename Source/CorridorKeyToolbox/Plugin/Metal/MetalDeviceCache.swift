//
//  MetalDeviceCache.swift
//  Corridor Key Toolbox
//
//  Caches Metal devices, command queues, and compiled pipeline states so the
//  render path never does I/O. Compute pipelines are cached per device; render
//  pipelines (which depend on the destination pixel format) are cached per
//  (device, pixel format) pair. Final Cut Pro may hand us tiles from multiple
//  GPUs within the same session.
//

import Foundation
import Metal
import MetalPerformanceShaders

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
            return "Corridor Key Toolbox could not locate its compiled Metal library."
        case .missingShaderFunction(let name):
            return "Corridor Key Toolbox could not find Metal function \(name)."
        case .unknownDevice(let registryID):
            return "Corridor Key Toolbox was handed an unfamiliar GPU (registry id \(registryID))."
        case .queueExhausted:
            return "All Corridor Key Toolbox command queues are currently in flight."
        case .textureAllocationFailed:
            return "Corridor Key Toolbox could not allocate an intermediate Metal texture."
        case .commandBufferCreationFailed:
            return "Corridor Key Toolbox could not create a Metal command buffer."
        case .commandEncoderCreationFailed:
            return "Corridor Key Toolbox could not create a Metal command encoder."
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
    let greenHint: any MTLComputePipelineState
    let sourcePassthrough: any MTLComputePipelineState
    let applyScreenMatrix: any MTLComputePipelineState
    let resample: any MTLComputePipelineState
    let extractHint: any MTLComputePipelineState
    let refinerBlend: any MTLComputePipelineState
    let lightWrap: any MTLComputePipelineState
    let edgeDecontaminate: any MTLComputePipelineState
    let ccLabelInit: any MTLComputePipelineState
    let ccLabelPropagate: any MTLComputePipelineState
    let ccLabelPointerJump: any MTLComputePipelineState
    let ccLabelCount: any MTLComputePipelineState
    let ccLabelFilter: any MTLComputePipelineState

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
        greenHint = try compute("corridorKeyGreenHintKernel")
        sourcePassthrough = try compute("corridorKeySourcePassthroughKernel")
        applyScreenMatrix = try compute("corridorKeyApplyScreenMatrixKernel")
        resample = try compute("corridorKeyResampleKernel")
        extractHint = try compute("corridorKeyExtractHintKernel")
        refinerBlend = try compute("corridorKeyRefinerBlendKernel")
        lightWrap = try compute("corridorKeyLightWrapKernel")
        edgeDecontaminate = try compute("corridorKeyEdgeDecontaminateKernel")
        ccLabelInit = try compute("corridorKeyCCLabelInitKernel")
        ccLabelPropagate = try compute("corridorKeyCCLabelPropagateKernel")
        ccLabelPointerJump = try compute("corridorKeyCCLabelPointerJumpKernel")
        ccLabelCount = try compute("corridorKeyCCLabelCountKernel")
        ccLabelFilter = try compute("corridorKeyCCLabelFilterKernel")
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
        descriptor.label = "Corridor Key Toolbox Compose"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        compose = try device.makeRenderPipelineState(descriptor: descriptor)
    }
}

/// Per-device state: the shared Metal library, compiled compute pipelines,
/// a pool of command queues, a per-format cache of render pipelines, a
/// reusable intermediate-texture pool, and a small cache of Gaussian weight
/// buffers.
final class MetalDeviceCacheEntry {
    let device: any MTLDevice
    let library: any MTLLibrary
    let computePipelines: CorridorKeyComputePipelines

    /// Reusable intermediate textures. Callers acquire via `texturePool` and
    /// return either manually (tests) or via `PooledTexture.returnOnCompletion`
    /// registered against a command buffer (renders).
    let texturePool: IntermediateTexturePool

    private let queueLock = NSLock()
    private var commandQueues: [any MTLCommandQueue]
    private var availability: [Bool]

    private let renderPipelinesLock = NSLock()
    private var renderPipelines: [MTLPixelFormat: CorridorKeyRenderPipelines] = [:]

    private let weightsLock = NSLock()
    private var weightBuffers: [GaussianWeightsKey: any MTLBuffer] = [:]

    private let mpsLock = NSLock()
    private var gaussianBlurs: [MPSGaussianKey: MPSImageGaussianBlur] = [:]
    private var morphDilates: [MPSMorphologyKey: MPSImageDilate] = [:]
    private var morphErodes: [MPSMorphologyKey: MPSImageErode] = [:]
    private var lanczosScales: MPSImageLanczosScale?

    /// Signals completion of command buffers back to CPU-waiting callers
    /// without a busy-spin `waitUntilCompleted`. Every entry owns its own
    /// event object; monotonic `signalledValue` means no reset ever needed.
    let sharedEvent: any MTLSharedEvent
    private let sharedEventValueLock = NSLock()
    private var sharedEventNextValue: UInt64 = 1
    let sharedEventListener: MTLSharedEventListener

    convenience init(device: any MTLDevice) throws {
        guard let library = device.makeDefaultLibrary() else {
            throw MetalDeviceCacheError.missingDefaultLibrary
        }
        try self.init(device: device, library: library)
    }

    /// Dependency-injected init used by the SPM test target, which compiles
    /// the shader library from source at run-time instead of relying on the
    /// bundle-embedded `default.metallib` that Xcode produces for the FxPlug
    /// target. Production code should use the zero-argument convenience
    /// init.
    init(device: any MTLDevice, library: any MTLLibrary) throws {
        self.device = device
        self.library = library
        self.computePipelines = try CorridorKeyComputePipelines(device: device, library: library)
        self.texturePool = IntermediateTexturePool(device: device)

        let queueCount = 4
        var queues: [any MTLCommandQueue] = []
        queues.reserveCapacity(queueCount)
        for _ in 0..<queueCount {
            if let queue = device.makeCommandQueue() {
                queue.label = "Corridor Key Toolbox"
                queues.append(queue)
            }
        }
        self.commandQueues = queues
        self.availability = Array(repeating: true, count: queues.count)

        // Shared-event machinery: the listener runs signal callbacks on a
        // dedicated dispatch queue so completion handlers never run on the
        // calling render thread.
        guard let event = device.makeSharedEvent() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        self.sharedEvent = event
        self.sharedEventListener = MTLSharedEventListener(
            dispatchQueue: DispatchQueue(label: "corridorkey.mtlsharedevent", qos: .userInitiated)
        )
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
    ///
    /// Prefer `texturePool.acquire(...)` for render-hot paths — this method
    /// remains for call sites that want a one-shot texture (tests and a
    /// handful of analysis helpers).
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

    // MARK: - Gaussian blur weight cache

    /// Key for the Gaussian weights cache. `radius` is the kernel radius in
    /// taps; `sigmaTenths` is the sigma rounded to one decimal place so nearly
    /// identical blurs share a buffer.
    struct GaussianWeightsKey: Hashable, Sendable {
        let radius: Int
        let sigmaTenths: Int
    }

    /// Returns a shared MTLBuffer of normalised Gaussian weights for the
    /// supplied radius and sigma. Safe to call concurrently from multiple
    /// render threads.
    func gaussianWeightsBuffer(radius: Int, sigma: Float) -> (buffer: any MTLBuffer, count: Int)? {
        guard radius > 0 else { return nil }
        let clampedSigma = max(sigma, 0.01)
        let sigmaTenths = Int((clampedSigma * 10).rounded())
        let key = GaussianWeightsKey(radius: radius, sigmaTenths: sigmaTenths)

        weightsLock.lock()
        if let existing = weightBuffers[key] {
            weightsLock.unlock()
            return (existing, radius + 1)
        }
        weightsLock.unlock()

        let quantisedSigma = Float(max(sigmaTenths, 1)) * 0.1
        let weights = Self.makeGaussianWeights(radius: radius, sigma: quantisedSigma)
        guard let buffer = weights.withUnsafeBufferPointer({ pointer -> (any MTLBuffer)? in
            guard let base = pointer.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base,
                length: pointer.count * MemoryLayout<Float>.size,
                options: .storageModeShared
            )
        }) else {
            return nil
        }
        buffer.label = "CK Gaussian Weights r=\(radius) σ=\(quantisedSigma)"

        weightsLock.lock()
        // Another thread may have raced in and cached an equivalent buffer.
        // Keep the first one to land so subsequent lookups hit the same
        // object — two concurrent uses of `weights_of_same_radius` don't
        // visually differ but reusing lets the pool stay tight.
        if let existing = weightBuffers[key] {
            weightsLock.unlock()
            return (existing, radius + 1)
        }
        weightBuffers[key] = buffer
        weightsLock.unlock()
        return (buffer, radius + 1)
    }

    /// Computes a normalised 1-sided Gaussian kernel. Returns the centre tap
    /// plus `radius` one-sided taps — the shader mirrors the taps around the
    /// origin so the stored array has `radius + 1` entries.
    static func makeGaussianWeights(radius: Int, sigma: Float) -> [Float] {
        var weights = [Float]()
        weights.reserveCapacity(radius + 1)
        var total: Float = 0
        let twoSigmaSquared = 2 * sigma * sigma
        for index in 0...radius {
            let offset = Float(index)
            let weight = exp(-(offset * offset) / twoSigmaSquared)
            weights.append(weight)
            total += (index == 0) ? weight : weight * 2
        }
        if total > 0 {
            for index in weights.indices { weights[index] /= total }
        }
        return weights
    }

    // MARK: - MPS kernel caches

    struct MPSGaussianKey: Hashable, Sendable {
        let sigmaTenths: Int
    }

    struct MPSMorphologyKey: Hashable, Sendable {
        let kernelWidth: Int
        let kernelHeight: Int
    }

    /// Returns a cached `MPSImageGaussianBlur` for the requested sigma,
    /// creating one on first use. `sigma <= 0` returns `nil` — callers should
    /// fall back to the identity path.
    func mpsGaussianBlur(sigma: Float) -> MPSImageGaussianBlur? {
        guard sigma > 0 else { return nil }
        let sigmaTenths = max(Int((sigma * 10).rounded()), 1)
        let key = MPSGaussianKey(sigmaTenths: sigmaTenths)
        mpsLock.lock()
        if let existing = gaussianBlurs[key] {
            mpsLock.unlock()
            return existing
        }
        let quantisedSigma = Float(sigmaTenths) * 0.1
        let blur = MPSImageGaussianBlur(device: device, sigma: quantisedSigma)
        blur.edgeMode = .clamp
        gaussianBlurs[key] = blur
        mpsLock.unlock()
        return blur
    }

    /// Returns a cached `MPSImageDilate` of the supplied side length.
    /// Size is always odd to keep the kernel centred; even values are
    /// rounded up.
    func mpsDilate(kernelSide: Int) -> MPSImageDilate? {
        let side = Self.roundedToOdd(kernelSide)
        guard side > 1 else { return nil }
        let key = MPSMorphologyKey(kernelWidth: side, kernelHeight: side)
        mpsLock.lock()
        if let existing = morphDilates[key] {
            mpsLock.unlock()
            return existing
        }
        let values = [Float](repeating: 0, count: side * side)
        let dilate = MPSImageDilate(
            device: device,
            kernelWidth: side,
            kernelHeight: side,
            values: values
        )
        dilate.edgeMode = .clamp
        morphDilates[key] = dilate
        mpsLock.unlock()
        return dilate
    }

    /// Returns a cached `MPSImageErode` of the supplied side length.
    func mpsErode(kernelSide: Int) -> MPSImageErode? {
        let side = Self.roundedToOdd(kernelSide)
        guard side > 1 else { return nil }
        let key = MPSMorphologyKey(kernelWidth: side, kernelHeight: side)
        mpsLock.lock()
        if let existing = morphErodes[key] {
            mpsLock.unlock()
            return existing
        }
        let values = [Float](repeating: 0, count: side * side)
        let erode = MPSImageErode(
            device: device,
            kernelWidth: side,
            kernelHeight: side,
            values: values
        )
        erode.edgeMode = .clamp
        morphErodes[key] = erode
        mpsLock.unlock()
        return erode
    }

    /// Returns the shared `MPSImageLanczosScale` for this device.
    func mpsLanczosScale() -> MPSImageLanczosScale {
        mpsLock.lock()
        if let existing = lanczosScales {
            mpsLock.unlock()
            return existing
        }
        let scaler = MPSImageLanczosScale(device: device)
        lanczosScales = scaler
        mpsLock.unlock()
        return scaler
    }

    private static func roundedToOdd(_ value: Int) -> Int {
        let clamped = max(value, 1)
        return clamped % 2 == 0 ? clamped + 1 : clamped
    }

    // MARK: - Shared-event value allocation

    /// Returns a new monotonically-increasing signal value for this entry's
    /// shared event. Use this to pair `commandBuffer.encodeSignalEvent(...)`
    /// with a matching `sharedEventListener.notify` callback so CPU-waiting
    /// callers never spin on `waitUntilCompleted`.
    func nextSharedEventValue() -> UInt64 {
        sharedEventValueLock.lock()
        defer { sharedEventValueLock.unlock() }
        let value = sharedEventNextValue
        sharedEventNextValue &+= 1
        return value
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
}
