//
//  TestHarness.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Shared setup for the Metal stage tests. Spins up a real `MTLDevice`,
//  compiles the Corridor Key shader library from the bundled source, and
//  prepares a `MetalDeviceCacheEntry` we can hand to `RenderStages`.
//
//  The shader library is compiled at run-time from the raw `.metal` file
//  rather than a pre-built `default.metallib` so SPM doesn't depend on
//  Xcode's Metal toolchain at test time — we just need the shader source
//  and a working Metal device, both of which any Apple Silicon Mac has.
//

import Foundation
import Metal
@testable import CorridorKeyToolboxMetalStages

/// Skippable failure used when the host has no Metal device (CI containers
/// occasionally disable the GPU). Tests catch this and early-return.
struct MetalUnavailable: Error, CustomStringConvertible {
    let reason: String
    var description: String { reason }
}

enum TestHarness {

    /// Creates a fresh `MetalDeviceCacheEntry` backed by the system default
    /// Metal device and the compiled Corridor Key shader library.
    static func makeEntry() throws -> MetalDeviceCacheEntry {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalUnavailable(reason: "No Metal device available on this host.")
        }
        let library = try makeShaderLibrary(device: device)
        return try MetalDeviceCacheEntry(device: device, library: library)
    }

    /// Convenience wrapper that surfaces `throws` to the test runtime and
    /// skips instead of failing when the hardware is absent.
    static func makeEntryOrSkip() throws -> MetalDeviceCacheEntry {
        do {
            return try makeEntry()
        } catch let error as MetalUnavailable {
            throw XCTSkipError(error.description)
        }
    }

    /// Reads the shader file (bundled as a resource alongside this target)
    /// plus the shared header, concatenates them, and hands the resulting
    /// string to `device.makeLibrary(source:)`. The header include in the
    /// shader is rewritten to a pair of raw struct definitions so the
    /// Metal compiler doesn't need to locate the `.h` file on disk.
    private static func makeShaderLibrary(device: any MTLDevice) throws -> any MTLLibrary {
        let bundle = Bundle.module
        guard let shaderURL = bundle.url(forResource: "CorridorKeyShaders", withExtension: "metal") else {
            throw MetalUnavailable(reason: "Could not find CorridorKeyShaders.metal in the test bundle.")
        }
        guard let headerURL = bundle.url(forResource: "CorridorKeyShaderTypes", withExtension: "h") else {
            throw MetalUnavailable(reason: "Could not find CorridorKeyShaderTypes.h in the test bundle.")
        }
        let shaderSource = try String(contentsOf: shaderURL, encoding: .utf8)
        let headerSource = try String(contentsOf: headerURL, encoding: .utf8)

        // Splice the header directly into the source so Metal's compiler
        // doesn't hunt the filesystem. The runtime `makeLibrary(source:)`
        // path doesn't support include paths out of the box, which is why
        // Xcode's build system prefers `-I` flags we can't hand to SPM.
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

/// Minimal `XCTSkipError` stand-in — Swift Testing has its own skip
/// mechanism (`#expect(...)` with `.disabled(...)`), but we lean on
/// early-return with a readable message because many tests touch the same
/// `TestHarness.makeEntryOrSkip()` helper.
struct XCTSkipError: Error, CustomStringConvertible {
    let message: String
    init(_ message: String) { self.message = message }
    var description: String { "Skipped: \(message)" }
}
