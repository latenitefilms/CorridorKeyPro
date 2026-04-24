//
//  main.swift
//  BenchmarkRunner
//
//  Headless regression tool for Corridor Key Toolbox. Reads a benchmark
//  clip folder (as shipped in `LLM Resources/Benchmark/`) and produces a
//  markdown report summarising matte-quality metrics that are sensitive to
//  the post-processing changes we'll land in subsequent phases.
//
//  The initial focus is *temporal* measurement because the upcoming
//  Temporal Coherence phase needs a concrete before/after. The tool is
//  designed so later phases can bolt on:
//
//  * Stage wall-time timings once we expose a public harness on
//    `CorridorKeyToolboxMetalStages`.
//  * Mean Absolute Error between two matte sequences for the Core ML
//    refiner parity test.
//  * Render-time benchmarks against the Metal stages for the kernel
//    fusion polish phase.
//
//  Run with:
//
//      swift run BenchmarkRunner \
//          --matte-folder "LLM Resources/Benchmark/NikoDruid/Matte EXR_DWAB" \
//          --output "Source/Benchmarks/baseline.md" \
//          --label "v1.0.0 Build 2 (baseline)"
//

import Foundation

/// Fall back to `Source/Benchmarks/<clip>.md` next to the package when the
/// user doesn't supply an explicit `--output`. Predictable location keeps
/// CI comparisons simple.
func defaultBenchmarkOutputURL(for matteFolder: URL) -> URL {
    let clipName = matteFolder.deletingLastPathComponent().lastPathComponent
    let cwd = URL(filePath: FileManager.default.currentDirectoryPath)
    return cwd
        .appending(path: "Benchmarks", directoryHint: .isDirectory)
        .appending(path: "\(clipName).md", directoryHint: .notDirectory)
}

do {
    let arguments = try BenchmarkArguments.parse(CommandLine.arguments)
    let sourceIdentifier = arguments.label ?? arguments.matteFolder.lastPathComponent

    print("Corridor Key Toolbox — BenchmarkRunner")
    print("Matte folder: \(arguments.matteFolder.path(percentEncoded: false))")
    print("Label       : \(sourceIdentifier)")

    let sequence = try MatteSequence(
        folder: arguments.matteFolder,
        frameLimit: arguments.frameLimit
    )

    guard let firstFrame = sequence.frames.first else {
        throw BenchmarkError.emptyFolder(arguments.matteFolder)
    }
    print("Frames       : \(sequence.frames.count)")
    print("Resolution   : \(firstFrame.width) x \(firstFrame.height)")

    print("Computing temporal metrics…")
    let clockStart = ContinuousClock.now
    let metrics = try await TemporalMetricsCalculator.compute(from: sequence)
    let elapsed = ContinuousClock.now - clockStart
    print("Completed in \(elapsed.formattedSeconds).")

    let report = BenchmarkReport(
        label: sourceIdentifier,
        matteFolder: arguments.matteFolder,
        frameCount: sequence.frames.count,
        frameWidth: firstFrame.width,
        frameHeight: firstFrame.height,
        temporalMetrics: metrics,
        runDuration: elapsed
    )

    let outputURL = arguments.outputURL ?? defaultBenchmarkOutputURL(for: arguments.matteFolder)
    try BenchmarkReportWriter.writeMarkdown(report, to: outputURL)
    print("Wrote report to \(outputURL.path(percentEncoded: false))")
} catch {
    fputs("BenchmarkRunner failed: \(error)\n", stderr)
    exit(1)
}

enum BenchmarkError: Error, CustomStringConvertible {
    case missingArgument(String)
    case unknownFlag(String)
    case emptyFolder(URL)
    case folderNotReadable(URL)
    case imageLoadFailed(URL, underlying: Error?)

    var description: String {
        switch self {
        case .missingArgument(let name):
            return "Missing required argument --\(name)"
        case .unknownFlag(let flag):
            return "Unknown flag \(flag)"
        case .emptyFolder(let url):
            return "No EXR frames found in \(url.path(percentEncoded: false))"
        case .folderNotReadable(let url):
            return "Folder is not readable: \(url.path(percentEncoded: false))"
        case .imageLoadFailed(let url, let underlying):
            if let underlying {
                return "Failed to load image at \(url.lastPathComponent): \(underlying)"
            }
            return "Failed to load image at \(url.lastPathComponent)"
        }
    }
}

struct BenchmarkArguments {
    var matteFolder: URL
    var outputURL: URL?
    var label: String?
    var frameLimit: Int?

    static func parse(_ rawArguments: [String]) throws -> BenchmarkArguments {
        var matteFolder: URL?
        var outputURL: URL?
        var label: String?
        var frameLimit: Int?

        var index = 1
        let arguments = rawArguments
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--matte-folder":
                guard index + 1 < arguments.count else {
                    throw BenchmarkError.missingArgument("matte-folder")
                }
                matteFolder = URL(filePath: arguments[index + 1])
                index += 2
            case "--output":
                guard index + 1 < arguments.count else {
                    throw BenchmarkError.missingArgument("output")
                }
                outputURL = URL(filePath: arguments[index + 1])
                index += 2
            case "--label":
                guard index + 1 < arguments.count else {
                    throw BenchmarkError.missingArgument("label")
                }
                label = arguments[index + 1]
                index += 2
            case "--frame-limit":
                guard index + 1 < arguments.count else {
                    throw BenchmarkError.missingArgument("frame-limit")
                }
                frameLimit = Int(arguments[index + 1])
                index += 2
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                throw BenchmarkError.unknownFlag(argument)
            }
        }

        guard let matteFolder else {
            throw BenchmarkError.missingArgument("matte-folder")
        }
        return BenchmarkArguments(
            matteFolder: matteFolder,
            outputURL: outputURL,
            label: label,
            frameLimit: frameLimit
        )
    }

    private static func printUsage() {
        let message = """
        Usage:
          swift run BenchmarkRunner \\
              --matte-folder <path-to-matte-sequence> \\
              [--output <markdown-path>] \\
              [--label <identifier>] \\
              [--frame-limit <n>]

        Reads an EXR matte sequence (single-channel float alpha) and reports
        temporal standard-deviation metrics as a markdown table. Designed as a
        quantitative before/after gate for post-processing changes.
        """
        print(message)
    }
}

extension Duration {
    /// Renders the duration as a fixed-precision seconds string. Foundation's
    /// `formatted` requires a locale we'd rather not depend on inside a CLI.
    var formattedSeconds: String {
        let attosecondsPerSecond: Int64 = 1_000_000_000_000_000_000
        let (seconds, attoseconds) = components
        let fractional = Double(attoseconds) / Double(attosecondsPerSecond)
        let total = Double(seconds) + fractional
        return total.formatted(.number.precision(.fractionLength(3))) + "s"
    }
}
