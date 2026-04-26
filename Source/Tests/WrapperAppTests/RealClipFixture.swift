//
//  RealClipFixture.swift
//  Corridor Key Toolbox — WrapperAppTests
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

    /// Source-tree path to the NikoDruid input clip. The Source
    /// directory sits next to `LLM Resources` in the repo, so we walk
    /// up from `#filePath` to find the repo root and append the
    /// well-known relative path.
    static func realClipURL(file: StaticString = #filePath) -> URL? {
        let here = URL(fileURLWithPath: String(describing: file))
        // Walk up until we find a sibling "LLM Resources" folder.
        var search = here.deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = search
                .appending(path: "LLM Resources/Benchmark/NikoDruid/Input.MP4")
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
