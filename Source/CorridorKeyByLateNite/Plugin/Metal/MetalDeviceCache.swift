//
//  MetalDeviceCache.swift
//  CorridorKey by LateNite
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
            return "CorridorKey by LateNite could not locate its compiled Metal library."
        case .missingShaderFunction(let name):
            return "CorridorKey by LateNite could not find Metal function \(name)."
        case .unknownDevice(let registryID):
            return "CorridorKey by LateNite was handed an unfamiliar GPU (registry id \(registryID))."
        case .queueExhausted:
            return "All CorridorKey by LateNite command queues are currently in flight."
        case .textureAllocationFailed:
            return "CorridorKey by LateNite could not allocate an intermediate Metal texture."
        case .commandBufferCreationFailed:
            return "CorridorKey by LateNite could not create a Metal command buffer."
        case .commandEncoderCreationFailed:
            return "CorridorKey by LateNite could not create a Metal command encoder."
        }
    }
}

/// Compute pipelines keyed only by device. They do not depend on the output
/// pixel format because compute shaders write untyped floats.
final class CorridorKeyComputePipelines: Sendable {
    let combineAndNormalize: any MTLComputePipelineState
    let normalizeToBuffer: any MTLComputePipelineState
    let alphaBufferToTexture: any MTLComputePipelineState
    let foregroundBufferToTexture: any MTLComputePipelineState
    let despill: any MTLComputePipelineState
    let alphaLevelsGamma: any MTLComputePipelineState
    let morphologyHorizontal: any MTLComputePipelineState
    let morphologyVertical: any MTLComputePipelineState
    let gaussianHorizontal: any MTLComputePipelineState
    let gaussianVertical: any MTLComputePipelineState
    /// Chroma-prior alpha-hint generator. Picks the screen channel
    /// from `CKChromaHintParams.screenColor`, so the same pipeline
    /// services both green and blue keys without needing a separate
    /// kernel per colour.
    let chromaHint: any MTLComputePipelineState
    /// Union-combiner for two alpha-hint textures. Used in Apple
    /// Vision hint mode to fold the chroma prior underneath the
    /// Vision subject mask so foreground props (which the subject
    /// detector ignores) still get marked as foreground.
    let hintUnion: any MTLComputePipelineState
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
    let matteRefineFused: any MTLComputePipelineState
    let foregroundPostProcess: any MTLComputePipelineState
    let temporalBlend: any MTLComputePipelineState
    let applyHintPoints: any MTLComputePipelineState

    init(device: any MTLDevice, library: any MTLLibrary) throws {
        func compute(_ name: String) throws -> any MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalDeviceCacheError.missingShaderFunction(name)
            }
            return try device.makeComputePipelineState(function: function)
        }
        combineAndNormalize = try compute("corridorKeyCombineAndNormalizeKernel")
        normalizeToBuffer = try compute("corridorKeyNormalizeToBufferKernel")
        alphaBufferToTexture = try compute("corridorKeyAlphaBufferToTextureKernel")
        foregroundBufferToTexture = try compute("corridorKeyForegroundBufferToTextureKernel")
        despill = try compute("corridorKeyDespillKernel")
        alphaLevelsGamma = try compute("corridorKeyAlphaLevelsGammaKernel")
        morphologyHorizontal = try compute("corridorKeyMorphologyHorizontalKernel")
        morphologyVertical = try compute("corridorKeyMorphologyVerticalKernel")
        gaussianHorizontal = try compute("corridorKeyGaussianHorizontalKernel")
        gaussianVertical = try compute("corridorKeyGaussianVerticalKernel")
        chromaHint = try compute("corridorKeyChromaHintKernel")
        hintUnion = try compute("corridorKeyHintUnionKernel")
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
        matteRefineFused = try compute("corridorKeyMatteRefineKernel")
        foregroundPostProcess = try compute("corridorKeyForegroundPostProcessKernel")
        temporalBlend = try compute("corridorKeyTemporalBlendKernel")
        applyHintPoints = try compute("corridorKeyApplyHintPointsKernel")
    }
}

/// Render pipelines depend on the destination pixel format (the colour
/// attachment format). The compose pipeline reads the per-frame intermediates
/// and writes the final RGBA to Final Cut Pro's destination texture.
final class CorridorKeyRenderPipelines: Sendable {
    let compose: any MTLRenderPipelineState
    /// OSC overlay pipeline. Same vertex shader as compose but a
    /// dedicated fragment shader that renders foreground / background
    /// hint dots with alpha blending so transparent regions let the
    /// underlying canvas show through.
    let drawOSC: any MTLRenderPipelineState

    init(device: any MTLDevice, library: any MTLLibrary, pixelFormat: MTLPixelFormat) throws {
        guard let vertexFunction = library.makeFunction(name: "corridorKeyComposeVertex") else {
            throw MetalDeviceCacheError.missingShaderFunction("corridorKeyComposeVertex")
        }
        guard let fragmentFunction = library.makeFunction(name: "corridorKeyComposeFragment") else {
            throw MetalDeviceCacheError.missingShaderFunction("corridorKeyComposeFragment")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "CorridorKey by LateNite Compose"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        compose = try device.makeRenderPipelineState(descriptor: descriptor)

        // OSC pipeline: shares the compose vertex shader (full-screen
        // quad with object-space UVs) but uses a dedicated fragment
        // that renders coloured discs with alpha blending against the
        // OSC's transparent destination canvas.
        guard let oscVertexFunction = library.makeFunction(name: "corridorKeyDrawOSCVertex") else {
            throw MetalDeviceCacheError.missingShaderFunction("corridorKeyDrawOSCVertex")
        }
        guard let oscFragmentFunction = library.makeFunction(name: "corridorKeyDrawOSCFragment") else {
            throw MetalDeviceCacheError.missingShaderFunction("corridorKeyDrawOSCFragment")
        }
        let oscDescriptor = MTLRenderPipelineDescriptor()
        oscDescriptor.label = "CorridorKey by LateNite OSC"
        oscDescriptor.vertexFunction = oscVertexFunction
        oscDescriptor.fragmentFunction = oscFragmentFunction
        oscDescriptor.colorAttachments[0].pixelFormat = pixelFormat
        // Standard "source over" alpha blending so transparent areas of
        // the OSC let the canvas show through.
        oscDescriptor.colorAttachments[0].isBlendingEnabled = true
        oscDescriptor.colorAttachments[0].rgbBlendOperation = .add
        oscDescriptor.colorAttachments[0].alphaBlendOperation = .add
        oscDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        oscDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        oscDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        oscDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        drawOSC = try device.makeRenderPipelineState(descriptor: oscDescriptor)
    }
}

/// Per-device state: the shared Metal library, compiled compute pipelines,
/// a pool of command queues, a per-format cache of render pipelines, a
/// reusable intermediate-texture pool, and a small cache of Gaussian weight
/// buffers. `@unchecked Sendable` because every mutable member is guarded
/// by an `NSLock` (`queueLock`, `renderPipelinesLock`, `weightsLock`,
/// `normalizedInputLock`, `mpsLock`, `sharedEventValueLock`).
final class MetalDeviceCacheEntry: @unchecked Sendable {
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
    /// High-water mark of simultaneously-in-flight command queues. Surfaced
    /// to `PluginLog` the first time the peak crosses the old (4-queue) cap
    /// so a saturation pattern in the wild gets recorded for follow-up.
    private var peakInFlightCount: Int = 0
    /// Tracks whether we've already logged the "saturation crossed 4"
    /// warning — keeps the log file from filling with the same line on
    /// every render tile in a saturated session.
    private var didLogSaturationWarning: Bool = false

    private let renderPipelinesLock = NSLock()
    private var renderPipelines: [MTLPixelFormat: CorridorKeyRenderPipelines] = [:]

    private let weightsLock = NSLock()
    private var weightBuffers: [GaussianWeightsKey: any MTLBuffer] = [:]

    private let normalizedInputLock = NSLock()
    private var normalizedInputBuffers: [Int: any MTLBuffer] = [:]

    private let mpsLock = NSLock()
    private var gaussianBlurs: [MPSGaussianKey: MPSImageGaussianBlur] = [:]
    private var morphDilates: [MPSMorphologyKey: MPSImageDilate] = [:]
    private var morphErodes: [MPSMorphologyKey: MPSImageErode] = [:]
    private var lanczosScales: MPSImageLanczosScale?

    /// Reusable `.shared`-storage staging texture used by the analysis-pass
    /// source readback path. Allocating a fresh 33 MB shared texture every
    /// frame (at the Maximum rung) piled up in the autorelease pool under
    /// Final Cut Pro's tight analyze-frame loop — memory grew unbounded
    /// until the driver fell behind. Caching a single texture per
    /// `(width, height, pixelFormat)` holds allocations to one per rung
    /// across the whole analysis.
    private struct AnalysisReadbackKey: Hashable {
        let width: Int
        let height: Int
        let pixelFormatRawValue: UInt
    }
    private let analysisReadbackLock = NSLock()
    private var analysisReadbackTextures: [AnalysisReadbackKey: any MTLTexture] = [:]

    /// Lazily-created Vision hint engine. Pinned to the cache entry so the
    /// CVMetalTextureCache and `VNGenerateForegroundInstanceMaskRequest`
    /// stay warm across analyse frames. `nil` means we tried and failed
    /// to create one — `visionHintEngine()` will not retry.
    private let visionHintLock = NSLock()
    private var visionHintEngineStorage: AnyObject?
    private var visionHintEngineFailedToInit: Bool = false

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

        // Eight queues per device. Empirically four was tight at 4K
        // 60-fps tiling — `borrowCommandQueue()` could return `nil` when
        // FCP dispatched faster than Metal committed, dropping the
        // tile. Doubling the pool removes that ceiling at negligible
        // cost (queues are cheap; pipeline state is shared across
        // them all).
        let queueCount = 8
        var queues: [any MTLCommandQueue] = []
        queues.reserveCapacity(queueCount)
        for _ in 0..<queueCount {
            if let queue = device.makeCommandQueue() {
                queue.label = "CorridorKey by LateNite"
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
            let inFlight = availability.lazy.filter { !$0 }.count
            if inFlight > peakInFlightCount {
                peakInFlightCount = inFlight
                if !didLogSaturationWarning, inFlight > 4 {
                    didLogSaturationWarning = true
                    PluginLog.warning(
                        "Command queue pool peak in-flight reached \(inFlight) — would have saturated the legacy 4-queue pool. Pool size is now 8 with telemetry tracking."
                    )
                }
            }
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

    /// Test / diagnostics hook returning the high-water mark of
    /// simultaneously-in-flight command queues observed since the
    /// cache entry was created. Safe to read at any time.
    func currentPeakInFlightCount() -> Int {
        queueLock.lock()
        defer { queueLock.unlock() }
        return peakInFlightCount
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

    // MARK: - Normalised-input buffer cache (zero-copy MLX)

    /// Returns a shared MTLBuffer sized for one rung's normalised NHWC
    /// tensor (`rung * rung * 4` floats). We cache one per rung because
    /// the plug-in rotates through at most five rungs (512/768/1024/
    /// 1536/2048) in a session, and each buffer is 4–67 MB — wastes a
    /// few hundred MB if we allocate fresh per frame, is near-zero
    /// steady state if we cache. `.storageModeShared` so
    /// `MLXArray(rawPointer:)` can read it without a copy on Apple
    /// Silicon's unified memory.
    func normalizedInputBuffer(forRung rung: Int) -> (any MTLBuffer)? {
        precondition(rung > 0, "rung must be positive")
        normalizedInputLock.lock()
        if let existing = normalizedInputBuffers[rung] {
            normalizedInputLock.unlock()
            return existing
        }
        normalizedInputLock.unlock()

        let byteCount = rung * rung * 4 * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        buffer.label = "CK Normalized Input \(rung)px"

        normalizedInputLock.lock()
        // Race: another thread may have just cached one. Keep theirs to
        // keep identity stable for callers.
        if let existing = normalizedInputBuffers[rung] {
            normalizedInputLock.unlock()
            return existing
        }
        normalizedInputBuffers[rung] = buffer
        normalizedInputLock.unlock()
        return buffer
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

    /// Encodes a tiny throwaway pass through every MPS kernel we use
    /// on the render hot path so MPS's internal Metal pipeline state
    /// is compiled before the first user-facing render. Without this,
    /// the first frame that hits MPS Lanczos / blur / morphology pays
    /// a ~50–100 ms compile stall on the render thread.
    ///
    /// Idempotent and safe to call from a background Task — every
    /// internal call is best-effort, failures are swallowed silently
    /// (the worst case is the lazy compile happens on first user
    /// render, which is exactly the legacy behaviour). Call once per
    /// entry, after `init`, from a low-priority queue.
    func prewarmMPSKernels() {
        guard !mpsPrewarmDone.swap(true) else { return }
        guard let queue = borrowCommandQueue() else { return }
        defer { returnCommandQueue(queue) }
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        commandBuffer.label = "CK MPS Prewarm"

        // 64×64 textures are small enough to stay off any meaningful
        // memory pressure path while still triggering MPS's encode-
        // time compile. Use `.rgba16Float` because that's the format
        // every shipping render path uses.
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: 64,
            height: 64,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        guard let source = device.makeTexture(descriptor: descriptor),
              let intermediate = device.makeTexture(descriptor: descriptor),
              let destination = device.makeTexture(descriptor: descriptor)
        else { return }

        let lanczos = mpsLanczosScale()
        lanczos.edgeMode = .clamp
        lanczos.encode(
            commandBuffer: commandBuffer,
            sourceTexture: source,
            destinationTexture: destination
        )

        // Warm Gaussian blur and morphology at small kernel sizes so
        // the pipeline state is JIT-compiled. We don't need every
        // sigma/radius variant — MPS shares pipeline state across
        // sizes once the underlying compute pipeline is warm.
        if let blur = mpsGaussianBlur(sigma: 2.0) {
            blur.encode(
                commandBuffer: commandBuffer,
                sourceTexture: source,
                destinationTexture: intermediate
            )
        }
        if let dilate = mpsDilate(kernelSide: 5) {
            dilate.encode(
                commandBuffer: commandBuffer,
                sourceTexture: intermediate,
                destinationTexture: destination
            )
        }
        if let erode = mpsErode(kernelSide: 5) {
            erode.encode(
                commandBuffer: commandBuffer,
                sourceTexture: destination,
                destinationTexture: intermediate
            )
        }

        commandBuffer.commit()
        // No `waitUntilCompleted` — the GPU work is genuinely
        // background; the next user-facing command buffer will
        // serialise behind us automatically through the command
        // queue. Keeping this fire-and-forget keeps the prewarm cost
        // off the calling thread.
    }

    /// Set on first call to `prewarmMPSKernels` so the work only runs
    /// once per entry. `OSAllocatedAtomic`-style swap via NSLock keeps
    /// the contention minimal in the rare case two render threads
    /// race to first-use.
    private let mpsPrewarmDone = OneShotLatch()

    private static func roundedToOdd(_ value: Int) -> Int {
        let clamped = max(value, 1)
        return clamped % 2 == 0 ? clamped + 1 : clamped
    }

    // MARK: - Analysis readback staging texture

    /// Returns a cached `.shared`-storage texture matching `(width, height,
    /// pixelFormat)` suitable as the destination of a blit from a private
    /// pool texture. Callers must not retain the texture across an analysis
    /// session; the cache is purged on `clearAnalysisReadbackTextures`.
    ///
    /// The returned texture is reused across frames, so callers are
    /// responsible for ensuring any outstanding GPU work that depends on
    /// it has completed before reissuing a blit into it (the analyser's
    /// blocking `commitAndWait` pattern meets this requirement).
    func analysisReadbackTexture(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat
    ) -> (any MTLTexture)? {
        let key = AnalysisReadbackKey(
            width: max(width, 1),
            height: max(height, 1),
            pixelFormatRawValue: pixelFormat.rawValue
        )
        analysisReadbackLock.lock()
        if let existing = analysisReadbackTextures[key] {
            analysisReadbackLock.unlock()
            return existing
        }
        analysisReadbackLock.unlock()

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: key.width,
            height: key.height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }
        texture.label = "CK Analysis Readback \(key.width)x\(key.height)"

        analysisReadbackLock.lock()
        // Race: another thread may have just cached one. Keep theirs so
        // identity stays stable.
        if let existing = analysisReadbackTextures[key] {
            analysisReadbackLock.unlock()
            return existing
        }
        analysisReadbackTextures[key] = texture
        analysisReadbackLock.unlock()
        return texture
    }

    /// Called at the end of an analysis session to release the staging
    /// textures. Re-warms lazily on the next analysis pass.
    func clearAnalysisReadbackTextures() {
        analysisReadbackLock.lock()
        analysisReadbackTextures.removeAll(keepingCapacity: false)
        analysisReadbackLock.unlock()
    }

    // MARK: - Cached-alpha decode buffer

    /// Per-(width, height) host-coherent MTLBuffer that the cached
    /// matte's vImage decode writes into. Cached because the user
    /// rotates through at most a handful of inference resolutions in a
    /// session; allocating fresh per render-from-cache frame would
    /// cost ~4 MB / allocation at 1024 and 67 MB at 2048, which the
    /// autorelease pool can't reclaim fast enough during scrubbing.
    private struct CachedAlphaBufferKey: Hashable {
        let width: Int
        let height: Int
    }
    private let cachedAlphaLock = NSLock()
    private var cachedAlphaBuffers: [CachedAlphaBufferKey: any MTLBuffer] = [:]

    func cachedAlphaBuffer(width: Int, height: Int) -> (any MTLBuffer)? {
        let key = CachedAlphaBufferKey(width: width, height: height)
        cachedAlphaLock.lock()
        if let existing = cachedAlphaBuffers[key] {
            cachedAlphaLock.unlock()
            return existing
        }
        cachedAlphaLock.unlock()

        let byteCount = width * height * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        buffer.label = "CK Cached Alpha \(width)x\(height)"
        cachedAlphaLock.lock()
        if let existing = cachedAlphaBuffers[key] {
            cachedAlphaLock.unlock()
            return existing
        }
        cachedAlphaBuffers[key] = buffer
        cachedAlphaLock.unlock()
        return buffer
    }

    // MARK: - Connected-components label-counts buffer

    /// Per-(width, height) device-private buffer used as the atomic
    /// counts target for the despeckle pass. Allocating fresh per
    /// frame costs 64 MB at 4K (one UInt32 per pixel), which the
    /// despeckle path was previously paying every render — caching
    /// removes the allocation from the hot path entirely. The blit
    /// fill that zeroes the buffer at the start of each pass remains,
    /// so semantics are unchanged from the per-frame allocation case.
    /// Storage mode is `.private` because no CPU read is required;
    /// the count + filter kernels both run GPU-only.
    private struct CCCountsBufferKey: Hashable {
        let width: Int
        let height: Int
    }
    private let ccCountsLock = NSLock()
    private var ccCountsBuffers: [CCCountsBufferKey: any MTLBuffer] = [:]

    func connectedComponentsCountsBuffer(width: Int, height: Int) -> (any MTLBuffer)? {
        let key = CCCountsBufferKey(width: width, height: height)
        ccCountsLock.lock()
        if let existing = ccCountsBuffers[key] {
            ccCountsLock.unlock()
            return existing
        }
        ccCountsLock.unlock()

        // `+ 2` matches the despeckle pass's labelCapacity: index 0 is
        // reserved for background, and labels are 1-indexed up to
        // `width * height` (inclusive) by the init kernel.
        let labelCapacity = width * height + 2
        let byteCount = labelCapacity * MemoryLayout<UInt32>.stride
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModePrivate) else {
            return nil
        }
        buffer.label = "CK CC Counts \(width)x\(height)"

        ccCountsLock.lock()
        if let existing = ccCountsBuffers[key] {
            ccCountsLock.unlock()
            return existing
        }
        ccCountsBuffers[key] = buffer
        ccCountsLock.unlock()
        return buffer
    }

    // MARK: - Vision hint engine

    /// Lazily creates and returns a `VisionHintEngine` for this device.
    /// Returns `nil` on macOS < 14 or when the texture cache could not
    /// be created. The engine is cached for the lifetime of the device
    /// entry so per-frame analyse calls don't pay the texture-cache
    /// setup cost. `AnyObject` storage avoids hard-linking the
    /// availability-gated type into this file's signature.
    func visionHintEngine() -> AnyObject? {
        if #available(macOS 14.0, *) {
            visionHintLock.lock()
            if let engine = visionHintEngineStorage {
                visionHintLock.unlock()
                return engine
            }
            if visionHintEngineFailedToInit {
                visionHintLock.unlock()
                return nil
            }
            visionHintLock.unlock()

            let engine: VisionHintEngine
            do {
                engine = try VisionHintEngine(cacheEntry: self)
            } catch {
                PluginLog.error("Vision hint engine init failed: \(error.localizedDescription)")
                visionHintLock.lock()
                visionHintEngineFailedToInit = true
                visionHintLock.unlock()
                return nil
            }

            visionHintLock.lock()
            // Race: keep the first-stored engine if another thread won.
            if let existing = visionHintEngineStorage {
                visionHintLock.unlock()
                return existing
            }
            visionHintEngineStorage = engine
            visionHintLock.unlock()
            return engine
        }
        return nil
    }

    /// Drops the Vision engine's compiled inference graph and texture
    /// cache. Called at the end of an analyse session so long editing
    /// sessions don't accumulate Vision state.
    func releaseVisionHintEngine() {
        if #available(macOS 14.0, *) {
            visionHintLock.lock()
            let engine = visionHintEngineStorage as? VisionHintEngine
            visionHintLock.unlock()
            engine?.releaseCachedResources()
        }
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

/// One-shot latch used by `prewarmMPSKernels` to ensure the warm-up
/// pass runs at most once per entry, even under racing first-use from
/// multiple render threads. Pure NSLock-backed bool so the type stays
/// available on every macOS version we support without dragging in
/// `Synchronization` for a single flag.
final class OneShotLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false

    /// Returns the previous value and unconditionally sets the flag.
    /// Callers branch on the *return*: a `true` result means "someone
    /// else already swapped" and the caller should skip its work.
    @discardableResult
    func swap(_ newValue: Bool) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let previous = fired
        fired = newValue
        return previous
    }
}

/// Singleton cache shared by every plug-in instance in the XPC service.
final class MetalDeviceCache: @unchecked Sendable {
    static let shared = MetalDeviceCache()

    private let entriesLock = NSLock()
    private var entries: [UInt64: MetalDeviceCacheEntry] = [:]

    func entry(for device: any MTLDevice) throws -> MetalDeviceCacheEntry {
        entriesLock.lock()
        let existing = entries[device.registryID]
        if let existing {
            entriesLock.unlock()
            return existing
        }
        let newEntry = try MetalDeviceCacheEntry(device: device)
        entries[device.registryID] = newEntry
        entriesLock.unlock()
        // First-touch MPS pre-warm runs off-thread so the calling
        // render path doesn't pay the ~50–100 ms MPS pipeline-compile
        // stall on its first frame. Best-effort; if the dispatched
        // queue is busy or the device rejects the small textures the
        // entry will lazily compile MPS state on first real use, the
        // same as before this hook landed.
        Task.detached(priority: .utility) {
            newEntry.prewarmMPSKernels()
        }
        return newEntry
    }

    func device(forRegistryID registryID: UInt64) -> (any MTLDevice)? {
        for device in MTLCopyAllDevices() where device.registryID == registryID {
            return device
        }
        return nil
    }
}
