//
//  MatteSequence.swift
//  BenchmarkRunner
//
//  Loads a directory of single-channel matte images (EXR or PNG) into a
//  lazy, ordered list. Does not hold decoded pixel data — each frame is
//  decoded on demand — so the calling code can stream a 500-frame 4K
//  sequence through temporal analysis without exhausting memory.
//

import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers

/// Sorted, lazy accessor over a directory of matte frames. The order comes
/// from `String.localizedStandardCompare`, which matches Finder's numeric
/// ordering so `Matte_00000001.exr` precedes `Matte_00000010.exr`.
struct MatteSequence {
    /// One entry per frame on disk. `width`/`height` are read from the first
    /// frame and every subsequent frame is required to match.
    struct Frame {
        let url: URL
        let width: Int
        let height: Int
    }

    let folder: URL
    let frames: [Frame]

    init(folder: URL, frameLimit: Int? = nil) throws {
        self.folder = folder

        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: folder,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ) else {
            throw BenchmarkError.folderNotReadable(folder)
        }

        var candidateURLs: [URL] = []
        for case let fileURL as URL in enumerator {
            if MatteSequence.isSupported(fileURL) {
                candidateURLs.append(fileURL)
            }
        }
        candidateURLs.sort { lhs, rhs in
            lhs.lastPathComponent.localizedStandardCompare(rhs.lastPathComponent) == .orderedAscending
        }

        if let frameLimit, candidateURLs.count > frameLimit {
            candidateURLs = Array(candidateURLs.prefix(frameLimit))
        }

        guard let firstURL = candidateURLs.first else {
            self.frames = []
            return
        }

        // Peek at the first image so downstream code can validate every
        // subsequent frame matches the same dimensions without re-reading.
        guard let referenceSize = MatteSequence.readDimensions(from: firstURL) else {
            throw BenchmarkError.imageLoadFailed(firstURL, underlying: nil)
        }

        self.frames = candidateURLs.map { url in
            Frame(url: url, width: referenceSize.width, height: referenceSize.height)
        }
    }

    /// Matte sequences ship as EXR in the benchmark folders, but we also
    /// accept PNG so the tool can be pointed at ad-hoc test content.
    private static func isSupported(_ url: URL) -> Bool {
        let suffix = url.pathExtension.lowercased()
        return suffix == "exr" || suffix == "png" || suffix == "tif" || suffix == "tiff"
    }

    private static func readDimensions(from url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
            return nil
        }
        let widthNumber = properties[kCGImagePropertyPixelWidth] as? NSNumber
        let heightNumber = properties[kCGImagePropertyPixelHeight] as? NSNumber
        guard let width = widthNumber?.intValue, let height = heightNumber?.intValue else {
            return nil
        }
        return (width, height)
    }
}
