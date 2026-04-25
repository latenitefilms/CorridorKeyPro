//
//  MLXKeyingEngine.swift
//  Corridor Key Toolbox
//
//  Loads a CorridorKey `.mlxfn` bridge via mlx-swift's public
//  `ImportedFunction` API and runs inference for one frame at a time. The
//  bridge file is a pre-compiled MLX graph produced upstream by the
//  CorridorKey training pipeline, so this engine is a thin adapter between
//  the render pipeline's Metal textures and MLX's tensor API.
//
//  Apple Silicon's unified memory means the Metal↔MLX hand-off is mostly a
//  pointer copy; the GPU work itself is scheduled by MLX against the Neural
//  Engine or the GPU as appropriate for the compiled graph.
//

import Foundation
import Metal
import MLX
import simd

/// Names of the bundled `.mlxfn` artefacts. Matches CorridorKey-Runtime's
/// `corridorkey_mlx_bridge_{N}.mlxfn` convention so the same Hugging Face
/// release can be used unmodified.
enum MLXBridgeArtifact {
    static let filenameStem = "corridorkey_mlx_bridge"

    /// Supported bridge resolutions, in preference order from lowest to
    /// highest. `closestSupportedResolution` walks this list.
    static let ladder: [Int] = [512, 768, 1024, 1536, 2048]

    static func filename(forResolution resolution: Int) -> String {
        "\(filenameStem)_\(resolution).mlxfn"
    }

    /// Returns the ladder rung that is at least as large as `requested`,
    /// falling back to the maximum if nothing larger exists.
    static func closestSupportedResolution(forRequested requested: Int) -> Int? {
        ladder.first(where: { $0 >= requested }) ?? ladder.last
    }
}

/// Lazy bundled-resource lookup that works from either the XPC service
/// bundle or its host app bundle, whichever contains the `.mlxfn` files.
private enum MLXBridgeResourceLocator {
    static func url(for filename: String) -> URL? {
        let fileManager = FileManager.default

        // Bundle.main.url(forResource:…) is the simplest path — it resolves
        // to the service bundle's Resources folder when FCP loads us.
        let filenameStem = (filename as NSString).deletingPathExtension
        let filenameExtension = (filename as NSString).pathExtension
        if let url = Bundle.main.url(forResource: filenameStem, withExtension: filenameExtension) {
            return url
        }

        // Also walk `Bundle.allBundles` in case the file lives inside the
        // wrapper app's Resources folder (for a dev build where the pluginkit
        // is copied in after Xcode bundles the mlxfn into the outer app).
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: filenameStem, withExtension: filenameExtension) {
                return url
            }
            if let resourceURL = bundle.resourceURL {
                let candidate = resourceURL.appending(path: filename)
                if fileManager.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        return nil
    }
}

final class MLXKeyingEngine: KeyingInferenceEngine, @unchecked Sendable {

    /// Strategy for handing the normalised input tensor to MLX. Switched at
    /// build time only — we are not (yet) confident enough in the
    /// rawPointer path to expose a runtime toggle.
    ///
    /// * `.zeroCopy` — the path introduced in commit `a4190e0`. Aliases the
    ///   shared MTLBuffer directly via `MLXArray(rawPointer:)`. Microbench
    ///   showed ~35 ms / ~285 MB savings per frame on the test rigs at the
    ///   time, but a real 4K clip on M1 Max measured 125–170 s per `eval()`
    ///   at the 2048 rung — wall-time inference far outside the warm-up's
    ///   25 s baseline. Held in reserve until we understand why.
    /// * `.cpuStaging` — the v1.0.0 build 1 behaviour. Reads the input
    ///   buffer into a reusable Swift `[Float]` and constructs an MLXArray
    ///   that owns its memory. MLX allocates internal storage the way it
    ///   wants, which seems to keep it on the fast path.
    private enum InputStrategy {
        case zeroCopy
        case cpuStaging
    }
    private static let inputStrategy: InputStrategy = .cpuStaging

    let backendDisplayName: String
    var guideSourceDescription: String

    private let cacheEntry: MetalDeviceCacheEntry
    /// Guards `importedFunction` / `loadedResolution` reads and writes. Held
    /// briefly so warm-up and per-frame renders never deadlock each other.
    private let stateLock = NSLock()
    /// Serialises the entire `run(...)` path. FxPlug calls us from multiple
    /// render threads concurrently and `ImportedFunction` is not documented as
    /// thread-safe — funnel everything through a single in-flight inference.
    private let runLock = NSLock()
    private var importedFunction: ImportedFunction?
    private var loadedResolution: Int = 0

    /// Reusable scratch buffer for the `.cpuStaging` strategy. Sized at
    /// warm-up so per-frame inference doesn't pay the 67 MB allocation
    /// cost on every call. `runLock` serialises access; no extra guard.
    private var inputScratch: [Float] = []

    init(cacheEntry: MetalDeviceCacheEntry) {
        self.cacheEntry = cacheEntry
        self.backendDisplayName = "MLX on \(cacheEntry.device.name)"
        self.guideSourceDescription = "Auto rough fallback"
        Self.applyMemoryLimitsOnce()
    }

    /// Caps MLX's internal buffer cache. The default is the device's
    /// recommended-max-working-set × 1.5 — on a 32 GB M1 Max that's
    /// ~30+ GB, and MLX happily fills it with buffers from intermediate
    /// computations across consecutive inferences. We measured 4.4 GB of
    /// cache after 30 sequential 512px inferences in the unit test
    /// `MLXMemoryTests`, scaling to ~70 GB at 2048 — which is exactly
    /// the 42 GB symptom Final Cut Pro hit during a 26-frame Analyse
    /// pass.
    ///
    /// 256 MiB is well above one inference's working set at every rung
    /// in the ladder (the largest, 2048, fits in ~120 MB of activations)
    /// while keeping memory bounded across long analyses. mlx-swift's
    /// own docs note that "many developers find that relatively small
    /// cache sizes (e.g. 2 MB) perform just as well" for inference; we
    /// pick a conservative ceiling that still leaves headroom for
    /// kernel JIT.
    ///
    /// Idempotent: the API is global to the MLX runtime, so running it
    /// once per process is sufficient. NSLock keeps it simple — the call
    /// only fires once per process so contention is negligible.
    private static let memoryLimitsLock = NSLock()
    private nonisolated(unsafe) static var memoryLimitsApplied = false
    private static func applyMemoryLimitsOnce() {
        memoryLimitsLock.lock()
        if memoryLimitsApplied {
            memoryLimitsLock.unlock()
            return
        }
        memoryLimitsApplied = true
        memoryLimitsLock.unlock()

        let cacheLimitBytes = 256 * 1024 * 1024
        MLX.Memory.cacheLimit = cacheLimitBytes
        PluginLog.notice("MLX cache limit pinned at \(cacheLimitBytes / (1024 * 1024)) MB.")
    }

    func supports(resolution: Int) -> Bool {
        guard let rung = MLXBridgeArtifact.closestSupportedResolution(forRequested: resolution) else {
            return false
        }
        return MLXBridgeResourceLocator.url(for: MLXBridgeArtifact.filename(forResolution: rung)) != nil
    }

    func prepare(resolution: Int) async throws {
        guard let rung = MLXBridgeArtifact.closestSupportedResolution(forRequested: resolution),
              let bridgeURL = MLXBridgeResourceLocator.url(for: MLXBridgeArtifact.filename(forResolution: rung))
        else {
            throw KeyingInferenceError.modelUnavailable(
                "No MLX bridge file bundled for \(resolution)px."
            )
        }
        try await prepare(bridgeURL: bridgeURL, rung: rung)
    }

    /// Test entry point that loads the bridge from an explicit URL. Useful
    /// from SPM unit tests where the `.mlxfn` lives in the test target's
    /// resources bundle (which is not enumerated by `Bundle.allBundles`).
    /// Production callers should use `prepare(resolution:)`.
    func prepare(bridgeURL: URL, rung: Int) async throws {
        if alreadyLoaded(rung: rung) { return }

        PluginLog.notice("Loading MLX bridge from \(bridgeURL.path).")
        let function: ImportedFunction
        do {
            function = try ImportedFunction(url: bridgeURL)
        } catch {
            throw KeyingInferenceError.modelUnavailable(
                "MLX could not load \(bridgeURL.lastPathComponent): \(error.localizedDescription)"
            )
        }

        // Drive one zero-filled inference to trigger MLX's JIT compilation and
        // allocate the Metal buffer pool. Finishing this before we advertise
        // the engine as loaded means the first real render frame never pays a
        // multi-second stall while MLX compiles on demand.
        await warmJIT(function: function, rung: rung)
        storeFunction(function, rung: rung)
    }

    /// Runs a throwaway inference on a zero tensor so MLX compiles the graph
    /// and warms the Metal buffer cache. Any failure here is non-fatal — the
    /// real inference will surface the same error to the caller later.
    private func warmJIT(function: ImportedFunction, rung: Int) async {
        let warmupStart = Date()
        let zeros = [Float](repeating: 0, count: rung * rung * 4)
        let input = MLXArray(zeros, [1, rung, rung, 4])
        do {
            let outputs = try function(input)
            eval(outputs)
        } catch {
            PluginLog.error("MLX JIT warm-up failed (non-fatal): \(error.localizedDescription)")
            return
        }
        let elapsedSeconds = Date().timeIntervalSince(warmupStart)
        PluginLog.notice("MLX JIT warm-up finished in \(String(format: "%.2f", elapsedSeconds))s for \(rung)px.")
    }

    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        runLock.lock()
        defer { runLock.unlock() }

        let (function, rung) = loadedState()
        guard let function, rung > 0 else {
            throw KeyingInferenceError.modelUnavailable("MLX bridge not prepared.")
        }
        _ = request.rawSourceTexture
        let expectedBytes = rung * rung * 4 * MemoryLayout<Float>.size
        guard request.normalisedInputBuffer.length >= expectedBytes else {
            throw KeyingInferenceError.modelUnavailable(
                "MLX input buffer is \(request.normalisedInputBuffer.length) bytes; expected ≥ \(expectedBytes)."
            )
        }

        // Step 1: build the MLX input. Two strategies live here; see
        // `InputStrategy` for the trade-off and the wall-time data.
        let inputBuffer = request.normalisedInputBuffer
        let inputArray: MLXArray
        switch Self.inputStrategy {
        case .zeroCopy:
            inputArray = MLXArray(
                rawPointer: inputBuffer.contents(),
                [1, rung, rung, 4],
                dtype: .float32,
                finalizer: { _ = inputBuffer }
            )
        case .cpuStaging:
            let expectedCount = rung * rung * 4
            if inputScratch.count != expectedCount {
                inputScratch = [Float](repeating: 0, count: expectedCount)
            }
            // Copy 67 MB once into the Swift scratch. On Apple Silicon's
            // unified memory both source and destination are CPU-visible,
            // so this is just a memcpy with no GPU sync. Lets MLX own the
            // input layout end-to-end, which empirically keeps `eval()`
            // on the fast path.
            inputScratch.withUnsafeMutableBufferPointer { destination in
                guard let destinationBase = destination.baseAddress else { return }
                memcpy(destinationBase, inputBuffer.contents(), expectedCount * MemoryLayout<Float>.size)
            }
            inputArray = MLXArray(inputScratch, [1, rung, rung, 4])
        }

        // Step 2: invoke the imported function. The graph returns
        // `(alpha, foreground)` per CorridorKey's bridge exporter.
        let results: [MLXArray]
        do {
            results = try function(inputArray)
        } catch {
            throw KeyingInferenceError.modelUnavailable(
                "MLX apply failed: \(error.localizedDescription)"
            )
        }
        guard results.count >= 2 else {
            throw KeyingInferenceError.modelUnavailable(
                "MLX bridge returned \(results.count) outputs; expected 2."
            )
        }
        try assertOutputShapes(alpha: results[0], foreground: results[1], rung: rung)

        // Step 3: force evaluation so the backing buffers are populated.
        eval(results[0], results[1])

        // Step 4: alias MLX's own output storage as MTLBuffers (still on
        // the GPU, no CPU copy) and encode a compute pass that reads them
        // into the output textures the caller provided. A completion
        // handler retains the MLXArrays so their backing memory survives
        // until the kernel finishes reading.
        let device = cacheEntry.device
        guard let alphaMLXBuffer = results[0].asMTLBuffer(device: device, noCopy: true) else {
            throw KeyingInferenceError.modelUnavailable("MLX alpha output could not be exposed as MTLBuffer.")
        }
        guard let foregroundMLXBuffer = results[1].asMTLBuffer(device: device, noCopy: true) else {
            throw KeyingInferenceError.modelUnavailable("MLX foreground output could not be exposed as MTLBuffer.")
        }

        guard let commandQueue = cacheEntry.borrowCommandQueue() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        defer { cacheEntry.returnCommandQueue(commandQueue) }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        commandBuffer.label = "Corridor Key Toolbox MLX Writeback"

        try RenderStages.writeAlphaBufferToTexture(
            buffer: alphaMLXBuffer,
            destination: output.alphaTexture,
            entry: cacheEntry,
            commandBuffer: commandBuffer
        )
        try RenderStages.writeForegroundBufferToTexture(
            buffer: foregroundMLXBuffer,
            destination: output.foregroundTexture,
            entry: cacheEntry,
            commandBuffer: commandBuffer
        )

        // Retain the MLXArrays (and the MTLBuffer aliases they back) until
        // the GPU is done reading them. `asMTLBuffer(noCopy:true)` requires
        // the MLXArray to outlive the MTLBuffer; `addCompletedHandler`
        // fires after the kernels are done, so keeping a strong
        // reference here is the simplest lifetime extension that works.
        // The captures cross a `@Sendable` boundary — MLXArray and
        // MTLBuffer aren't Sendable-typed, but they're safe to capture
        // because we only use them to keep allocations alive, not mutate
        // them. Wrap in a small `@unchecked Sendable` box to satisfy the
        // concurrency checker.
        final class RetainBox: @unchecked Sendable {
            let alphaArray: MLXArray
            let foregroundArray: MLXArray
            let alphaBuffer: any MTLBuffer
            let foregroundBuffer: any MTLBuffer
            init(alphaArray: MLXArray, foregroundArray: MLXArray, alphaBuffer: any MTLBuffer, foregroundBuffer: any MTLBuffer) {
                self.alphaArray = alphaArray
                self.foregroundArray = foregroundArray
                self.alphaBuffer = alphaBuffer
                self.foregroundBuffer = foregroundBuffer
            }
        }
        let retainBox = RetainBox(
            alphaArray: results[0],
            foregroundArray: results[1],
            alphaBuffer: alphaMLXBuffer,
            foregroundBuffer: foregroundMLXBuffer
        )
        commandBuffer.addCompletedHandler { _ in
            _ = retainBox
        }

        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()
        if let error = commandBuffer.error { throw error }
    }

    // MARK: - Lock-guarded state helpers

    private func alreadyLoaded(rung: Int) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return importedFunction != nil && loadedResolution == rung
    }

    private func loadedState() -> (ImportedFunction?, Int) {
        stateLock.lock()
        defer { stateLock.unlock() }
        return (importedFunction, loadedResolution)
    }

    private func storeFunction(_ function: ImportedFunction, rung: Int) {
        stateLock.lock()
        defer { stateLock.unlock() }
        if importedFunction != nil, loadedResolution == rung { return }
        importedFunction = function
        loadedResolution = rung
    }

    /// Validates that the loaded bridge's outputs have the expected NHWC layout.
    /// Fails loudly on mismatch so a misbuilt or misplaced `.mlxfn` doesn't
    /// silently corrupt downstream writes.
    private func assertOutputShapes(alpha: MLXArray, foreground: MLXArray, rung: Int) throws {
        let expectedAlpha = [1, rung, rung, 1]
        let expectedForeground = [1, rung, rung, 3]
        if alpha.shape != expectedAlpha || foreground.shape != expectedForeground {
            PluginLog.error(
                "MLX bridge returned unexpected shapes: alpha=\(alpha.shape) foreground=\(foreground.shape); expected \(expectedAlpha) and \(expectedForeground)."
            )
            throw KeyingInferenceError.modelUnavailable(
                "MLX bridge returned unexpected output shapes."
            )
        }
    }

}
