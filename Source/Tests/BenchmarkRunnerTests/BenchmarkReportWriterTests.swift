//
//  BenchmarkReportWriterTests.swift
//  BenchmarkRunnerTests
//
//  Assert that the markdown renderer includes the metrics we intend to
//  compare between runs. The exact layout is free to evolve so long as
//  these fixed-label lines keep working for CI-friendly diffs.
//

import Testing
import Foundation
@testable import BenchmarkRunner

@Suite struct BenchmarkReportWriterTests {

    @Test func renderedMarkdownContainsAllKeyMetrics() {
        let metrics = TemporalMetrics(
            globalMeanStdDev: 0.0123456,
            edgeBandMeanStdDev: 0.0789012,
            edgeBandPixelPercentage: 12.5,
            edgeBandP95StdDev: 0.1234,
            edgeBandMaxStdDev: 0.567,
            globalMeanAlpha: 0.41,
            frameCount: 120,
            width: 3840,
            height: 2160
        )
        let report = BenchmarkReport(
            label: "v1.0.0 Build 2 (baseline)",
            matteFolder: URL(filePath: "/tmp/fake"),
            frameCount: metrics.frameCount,
            frameWidth: metrics.width,
            frameHeight: metrics.height,
            temporalMetrics: metrics,
            runDuration: .seconds(3.75)
        )
        let markdown = BenchmarkReportWriter.render(report)

        #expect(markdown.contains("Corridor Key Toolbox — Matte Benchmark"))
        #expect(markdown.contains("v1.0.0 Build 2 (baseline)"))
        #expect(markdown.contains("| Frames | 120 |"))
        #expect(markdown.contains("3,840 × 2,160") || markdown.contains("3840 × 2160"))
        #expect(markdown.contains("Global mean σ"))
        #expect(markdown.contains("Edge-band mean σ"))
        #expect(markdown.contains("Edge-band p95 σ"))
        #expect(markdown.contains("Edge-band max σ"))
        #expect(markdown.contains("Edge-band pixel share"))
    }

    @Test func renderedMarkdownEndsWithTrailingNewline() {
        let metrics = TemporalMetrics(
            globalMeanStdDev: 0,
            edgeBandMeanStdDev: 0,
            edgeBandPixelPercentage: 0,
            edgeBandP95StdDev: 0,
            edgeBandMaxStdDev: 0,
            globalMeanAlpha: 0,
            frameCount: 0,
            width: 0,
            height: 0
        )
        let report = BenchmarkReport(
            label: "empty",
            matteFolder: URL(filePath: "/tmp/empty"),
            frameCount: 0,
            frameWidth: 0,
            frameHeight: 0,
            temporalMetrics: metrics,
            runDuration: .seconds(0)
        )
        let markdown = BenchmarkReportWriter.render(report)
        #expect(markdown.hasSuffix("\n"))
    }
}
