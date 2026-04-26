//
//  RealClipFixture.swift
//  CorridorKey by LateNite — WrapperAppTests
//
//  Locator and gate for the real NikoDruid green-screen clip the
//  benchmark suite uses. The file is 130 MB so we deliberately don't
//  bundle it inside the .xctest bundle — instead each test that needs
//  it reads it directly from the source-tree path. Tests that need it
//  call `try realClipURL()`; if the file is missing (e.g. a sparse
//  checkout), they early-out via `requireRealClip()` so the rest of
//  the suite still runs.
//

import Foundation
import Testing

enum RealClipFixture {

    /// Source-tree path to the full 130 MB NikoDruid input clip.
    /// Used by the parity tests that need a long clip to exercise
    /// the analyse pass over many frames.
    static func realClipURL(file: StaticString = #filePath) -> URL? {
        locate(relativePath: "LLM Resources/Benchmark/NikoDruid/Input.MP4", from: file)
    }

    /// Source-tree path to the 4-frame ProRes 4444 fixture used by
    /// the round-trip end-to-end tests. Much smaller than `Input.MP4`
    /// (~20 MB instead of 130 MB) and only four frames long, so the
    /// full analyse → render → export cycle finishes in seconds.
    static func fourFrameClipURL(file: StaticString = #filePath) -> URL? {
        locate(relativePath: "LLM Resources/Benchmark/NikoDruid/Niko - 4 frames.mov", from: file)
    }

    private static func locate(relativePath: String, from file: StaticString) -> URL? {
        let here = URL(fileURLWithPath: String(describing: file))
        var search = here.deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = search.appending(path: relativePath)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            let parent = search.deletingLastPathComponent()
            if parent == search { break }
            search = parent
        }
        return nil
    }

    /// `try` form — emits a `SkipTestError` so the test suite keeps
    /// running on machines / sparse checkouts where the file is
    /// absent. Combined with `#expect(throws:)` this is awkward; for
    /// most tests prefer `#require(realClipURL())` directly.
    enum FixtureMissing: Error, CustomStringConvertible {
        case notFound

        var description: String {
            "The NikoDruid Input.MP4 fixture is not present at "
            + "LLM Resources/Benchmark/NikoDruid/Input.MP4 — Real-clip "
            + "tests skipped."
        }
    }
}
