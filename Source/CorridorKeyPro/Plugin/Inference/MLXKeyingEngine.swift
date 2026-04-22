//
//  MLXKeyingEngine.swift
//  Corridor Key Pro
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
    let backendDisplayName: String
    var guideSourceDescription: String

    private let cacheEntry: MetalDeviceCacheEntry
    private let stateLock = NSLock()
    private var importedFunction: ImportedFunction?
    private var loadedResolution: Int = 0

    private var inputBuffer: [Float] = []

    init(cacheEntry: MetalDeviceCacheEntry) {
        self.cacheEntry = cacheEntry
        self.backendDisplayName = "MLX on \(cacheEntry.device.name)"
        self.guideSourceDescription = "Auto rough fallback"
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
        storeFunction(function, rung: rung)
    }

    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        let (function, rung) = loadedState()
        guard let function, rung > 0 else {
            throw KeyingInferenceError.modelUnavailable("MLX bridge not prepared.")
        }

        _ = request.rawSourceTexture
        // Step 1: stage the normalised tensor off the GPU into our CPU scratch
        // buffer. Apple Silicon's unified memory keeps this near-zero cost.
        try readNormalisedInput(texture: request.normalisedInputTexture, rung: rung)

        // Step 2: hand the tensor to MLX as an `MLXArray` and invoke the
        // imported function. The graph returns `(alpha, foreground)` per
        // CorridorKey's bridge exporter.
        let inputArray = MLXArray(inputBuffer, [1, rung, rung, 4])
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

        // Step 3: force evaluation so the backing buffers are populated.
        eval(results[0], results[1])

        // Step 4: copy the results into the shared-storage Metal textures the
        // render pipeline passed us.
        let alphaValues = results[0].asArray(Float.self)
        try writeScalarBuffer(
            buffer: alphaValues,
            texture: output.alphaTexture
        )

        let foregroundValues = results[1].asArray(Float.self)
        try writeForegroundBuffer(
            buffer: foregroundValues,
            texture: output.foregroundTexture
        )
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
        inputBuffer = Array(repeating: 0, count: rung * rung * 4)
    }

    // MARK: - Metal ↔ CPU staging

    /// Reads the supplied texture's pixels into `inputBuffer`. Assumes the
    /// caller has already dispatched a blit/synchronise so the contents are
    /// visible to the CPU — the render pipeline's `preCommandBuffer` does this
    /// before calling us.
    private func readNormalisedInput(texture: any MTLTexture, rung: Int) throws {
        let expected = rung * rung * 4
        if inputBuffer.count != expected {
            inputBuffer = Array(repeating: 0, count: expected)
        }

        let bytesPerRow = rung * 4 * MemoryLayout<Float>.size
        let region = MTLRegionMake2D(0, 0, rung, rung)
        inputBuffer.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.getBytes(
                    base,
                    bytesPerRow: bytesPerRow,
                    from: region,
                    mipmapLevel: 0
                )
            }
        }
    }

    /// Writes a tightly-packed single-channel Float32 buffer into an `.r32Float`
    /// destination. Matches the MLX output for alpha.
    private func writeScalarBuffer(
        buffer: [Float],
        texture: any MTLTexture
    ) throws {
        let width = texture.width
        let height = texture.height
        guard buffer.count >= width * height else {
            throw KeyingInferenceError.modelUnavailable(
                "MLX alpha buffer was \(buffer.count) floats; expected \(width * height)."
            )
        }
        let region = MTLRegionMake2D(0, 0, width, height)
        let bytesPerRow = width * MemoryLayout<Float>.size
        buffer.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.replace(region: region, mipmapLevel: 0, withBytes: base, bytesPerRow: bytesPerRow)
            }
        }
    }

    /// Expands a tightly-packed RGB Float32 buffer into an `.rgba32Float`
    /// destination by appending an opaque alpha channel.
    private func writeForegroundBuffer(
        buffer: [Float],
        texture: any MTLTexture
    ) throws {
        let width = texture.width
        let height = texture.height
        let pixelCount = width * height
        guard buffer.count >= pixelCount * 3 else {
            throw KeyingInferenceError.modelUnavailable(
                "MLX foreground buffer was \(buffer.count) floats; expected \(pixelCount * 3)."
            )
        }
        var rgba = [Float](repeating: 0, count: pixelCount * 4)
        for index in 0..<pixelCount {
            rgba[index * 4 + 0] = buffer[index * 3 + 0]
            rgba[index * 4 + 1] = buffer[index * 3 + 1]
            rgba[index * 4 + 2] = buffer[index * 3 + 2]
            rgba[index * 4 + 3] = 1
        }

        let region = MTLRegionMake2D(0, 0, width, height)
        let bytesPerRow = width * 4 * MemoryLayout<Float>.size
        rgba.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                texture.replace(region: region, mipmapLevel: 0, withBytes: base, bytesPerRow: bytesPerRow)
            }
        }
    }
}
