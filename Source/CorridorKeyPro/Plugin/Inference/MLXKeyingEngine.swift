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
//  pointer copy; the GPU work itself is scheduled by MLX.
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
    /// falling back to the maximum if nothing larger exists. Used by the
    /// coordinator when the user-selected quality rung does not have a
    /// matching bridge file.
    static func closestSupportedResolution(forRequested requested: Int) -> Int? {
        ladder.first(where: { $0 >= requested }) ?? ladder.last
    }
}

/// Lazy bundled-resource lookup that works from either the XPC service
/// bundle or its host app bundle, whichever contains the `.mlxfn` files.
private enum MLXBridgeResourceLocator {
    static func url(for filename: String) -> URL? {
        let fileManager = FileManager.default
        let candidateBundles = [Bundle.main] + Bundle.allBundles
        let searched = Set(candidateBundles.compactMap(\.bundleURL.standardizedFileURL))

        for bundleURL in searched {
            let direct = bundleURL.appendingPathComponent(filename)
            if fileManager.fileExists(atPath: direct.path) {
                return direct
            }
            let resourcesURL = bundleURL.appendingPathComponent("Contents/Resources/\(filename)")
            if fileManager.fileExists(atPath: resourcesURL.path) {
                return resourcesURL
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

    /// Scratch CPU buffers reused across frames so that per-frame inference
    /// does not churn the heap. Sized to the loaded resolution.
    private var inputBuffer: [Float] = []
    private var alphaBuffer: [Float] = []
    private var foregroundBuffer: [Float] = []

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

        // Load the compiled MLX graph outside the lock so that the lock is
        // only held for the cheap state swap — we never block on I/O or
        // kernel warm-up while holding it.
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

        // Step 1: stage the normalised tensor off the GPU into our CPU scratch
        // buffer. Apple Silicon's unified memory keeps this near-zero cost.
        try readNormalisedInput(into: &inputBuffer, texture: request.normalisedInputTexture, rung: rung)

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

        // Step 3: force evaluation so the backing buffers are populated, then
        // copy the floats out into our scratch arrays.
        var alphaOut = results[0]
        var foregroundOut = results[1]
        eval(alphaOut, foregroundOut)
        alphaBuffer = alphaOut.asArray(Float.self)
        foregroundBuffer = foregroundOut.asArray(Float.self)

        // Step 4: upload the results into the Metal textures the render
        // pipeline passed us so the rest of the pipeline can operate on them.
        try writeBufferToTexture(
            buffer: alphaBuffer,
            width: rung,
            height: rung,
            channels: 1,
            texture: output.alphaTexture
        )
        try writeBufferToTexture(
            buffer: foregroundBuffer,
            width: rung,
            height: rung,
            channels: 3,
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
        alphaBuffer = Array(repeating: 0, count: rung * rung)
        foregroundBuffer = Array(repeating: 0, count: rung * rung * 3)
    }

    // MARK: - Metal ↔ CPU staging

    /// Copies the supplied texture's pixels into `buffer`. Assumes the caller
    /// has already dispatched a blit/synchronise so the texture contents are
    /// visible to the CPU.
    private func readNormalisedInput(
        into buffer: inout [Float],
        texture: any MTLTexture,
        rung: Int
    ) throws {
        let bytesPerRow = rung * 4 * MemoryLayout<Float>.size
        let expected = rung * rung * 4
        precondition(buffer.count == expected, "Input scratch buffer sized incorrectly.")

        let region = MTLRegionMake2D(0, 0, rung, rung)
        buffer.withUnsafeMutableBufferPointer { pointer in
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

    /// Writes a float32 buffer back out to a destination Metal texture. The
    /// foreground path expands tightly-packed RGB into RGBA to match the
    /// render pipeline's 4-channel intermediate textures.
    private func writeBufferToTexture(
        buffer: [Float],
        width: Int,
        height: Int,
        channels: Int,
        texture: any MTLTexture
    ) throws {
        let region = MTLRegionMake2D(0, 0, width, height)
        if channels == 1 {
            let bytesPerRow = width * MemoryLayout<Float>.size
            buffer.withUnsafeBufferPointer { pointer in
                if let base = pointer.baseAddress {
                    texture.replace(region: region, mipmapLevel: 0, withBytes: base, bytesPerRow: bytesPerRow)
                }
            }
        } else {
            var rgba = [Float](repeating: 0, count: width * height * 4)
            for i in 0..<(width * height) {
                rgba[i * 4 + 0] = buffer[i * 3 + 0]
                rgba[i * 4 + 1] = buffer[i * 3 + 1]
                rgba[i * 4 + 2] = buffer[i * 3 + 2]
                rgba[i * 4 + 3] = 1.0
            }
            let bytesPerRow = width * 4 * MemoryLayout<Float>.size
            rgba.withUnsafeBufferPointer { pointer in
                if let base = pointer.baseAddress {
                    texture.replace(region: region, mipmapLevel: 0, withBytes: base, bytesPerRow: bytesPerRow)
                }
            }
        }
    }
}
