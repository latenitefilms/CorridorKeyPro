//
//  InferenceTestHarness.swift
//  CorridorKeyToolboxInferenceTests
//
//  Builds a usable `MetalDeviceCacheEntry` for the SPM inference tests.
//  Mirrors the `TestHarness` shape used by the Metal-stages tests but is
//  duplicated here so the inference target can run independently of the
//  Metal-stages test bundle.
//

import Foundation
import Metal
@testable import CorridorKeyToolboxMetalStages

enum InferenceTestHarness {

    /// Skippable failure for CI runners without a Metal device.
    struct MetalUnavailable: Error, CustomStringConvertible {
        let reason: String
        var description: String { reason }
    }

    /// Returns a fully-initialised entry whose Metal library is compiled
    /// from the bundled `CorridorKeyShaders.metal` source. Throws
    /// `MetalUnavailable` when the host has no GPU or the shaders fail to
    /// compile so tests can early-return cleanly.
    static func makeEntry() throws -> MetalDeviceCacheEntry {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalUnavailable(reason: "No Metal device available on this host.")
        }
        let library = try makeShaderLibrary(device: device)
        return try MetalDeviceCacheEntry(device: device, library: library)
    }

    /// Locates the bundled 512px MLX bridge inside the test target's
    /// resources bundle. `MLXBridgeResourceLocator` walks `Bundle.main` /
    /// `Bundle.allBundles`, neither of which reliably picks up SPM test
    /// resource bundles, so tests pass the URL explicitly.
    static func bridgeURL512() throws -> URL {
        guard let url = Bundle.module.url(
            forResource: "corridorkey_mlx_bridge_512",
            withExtension: "mlxfn"
        ) else {
            throw MetalUnavailable(reason: "corridorkey_mlx_bridge_512.mlxfn missing from test bundle.")
        }
        return url
    }

    /// Compiles the bundled shader source at run-time so the SPM build
    /// doesn't need Xcode's `default.metallib` step. The Metal-stages
    /// `TestHarness` uses the same trick.
    private static func makeShaderLibrary(device: any MTLDevice) throws -> any MTLLibrary {
        let bundle = Bundle.module
        guard let shaderURL = bundle.url(forResource: "CorridorKeyShaders", withExtension: "metal") else {
            // The shader source travels with `CorridorKeyToolboxMetalStages`,
            // not with the inference test target — fall back to its bundle.
            let metalBundle = MetalStagesResourceBundle.bundle
            guard let stagesShaderURL = metalBundle.url(forResource: "CorridorKeyShaders", withExtension: "metal") else {
                throw MetalUnavailable(reason: "Could not locate CorridorKeyShaders.metal.")
            }
            return try compileLibrary(device: device, shaderURL: stagesShaderURL, headerBundle: metalBundle)
        }
        return try compileLibrary(device: device, shaderURL: shaderURL, headerBundle: bundle)
    }

    private static func compileLibrary(
        device: any MTLDevice,
        shaderURL: URL,
        headerBundle: Bundle
    ) throws -> any MTLLibrary {
        guard let headerURL = headerBundle.url(forResource: "CorridorKeyShaderTypes", withExtension: "h") else {
            throw MetalUnavailable(reason: "Could not locate CorridorKeyShaderTypes.h.")
        }
        let shaderSource = try String(contentsOf: shaderURL, encoding: .utf8)
        let headerSource = try String(contentsOf: headerURL, encoding: .utf8)
        let combined = headerSource
            .replacing("#ifndef CorridorKeyShaderTypes_h", with: "")
            .replacing("#define CorridorKeyShaderTypes_h", with: "")
            .replacing("#endif /* CorridorKeyShaderTypes_h */", with: "")
            + "\n"
            + shaderSource.replacing(#"#include "CorridorKeyShaderTypes.h""#, with: "")
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        do {
            return try device.makeLibrary(source: combined, options: options)
        } catch {
            throw MetalUnavailable(reason: "Corridor Key shader compile failed: \(error.localizedDescription)")
        }
    }
}

/// Tiny shim that exposes `Bundle.module` from `CorridorKeyToolboxMetalStages`
/// so we can fish out the shared shader resources without making them
/// `public` on every consumer.
private enum MetalStagesResourceBundle {
    static var bundle: Bundle {
        // `Bundle.module` is module-scoped — referenced from a file that
        // belongs to `CorridorKeyToolboxMetalStages`, it returns that
        // module's resource bundle. We reach across via a public function
        // exposed there for exactly this purpose; if the symbol is
        // missing, we fall back to the test bundle and the shader-source
        // load will fail with a clean `MetalUnavailable`.
        MetalStagesResourceBundleAccessor.bundle()
    }
}
