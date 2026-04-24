// swift-tools-version: 6.0
//
//  Package.swift
//  Corridor Key Toolbox
//
//  Headless regression suite for the plug-in's pure-Swift logic and the
//  Metal stage helpers. The Xcode project remains the build system of
//  record for the FxPlug target itself; this Swift Package lets us unit-
//  test the pieces that don't depend on FxPlug so CI and local developer
//  loops can catch regressions without launching Final Cut Pro.
//
//  Two library targets:
//
//  * `CorridorKeyToolboxLogic` — pure Swift value types and codecs. Covers
//    the JSON/plist state contract, analysis snapshots, matte codec, and
//    pure colour math. Fast to compile, runnable on any arm64 macOS host.
//
//  * `CorridorKeyToolboxMetalStages` — brings in the Metal compute pool,
//    Gaussian weight cache, MPS-backed matte refiner, and the stage
//    helpers that drive the per-frame pipeline. Tests in
//    `CorridorKeyToolboxMetalStagesTests` feed synthetic textures through
//    each stage on a real `MTLDevice` and compare the results to
//    analytically expected output, so shader regressions never escape
//    CI silently.
//
//  Run with `swift test` from this directory.
//

import PackageDescription

let package = Package(
    name: "CorridorKeyToolboxLogic",
    defaultLocalization: "en",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "CorridorKeyToolboxLogic",
            targets: ["CorridorKeyToolboxLogic"]
        ),
        .library(
            name: "CorridorKeyToolboxMetalStages",
            targets: ["CorridorKeyToolboxMetalStages"]
        ),
        .executable(
            name: "BenchmarkRunner",
            targets: ["BenchmarkRunner"]
        )
    ],
    targets: [
        .target(
            name: "CorridorKeyToolboxLogic",
            path: "CorridorKeyToolbox/Plugin",
            // Keep the FxPlug-dependent files, the Xcode-managed resources,
            // and the shaders out of the SPM build — they only compile inside
            // the Xcode target.
            exclude: [
                "Info.plist",
                "CorridorKeyToolbox.entitlements",
                "PluginLog.swift",
                "main.swift",
                "CorridorKeyToolboxPlugIn.swift",
                "CorridorKeyToolboxPlugIn+Properties.swift",
                "CorridorKeyToolboxPlugIn+Render.swift",
                "CorridorKeyToolboxPlugIn+RenderRects.swift",
                "CorridorKeyToolboxPlugIn+PluginState.swift",
                "CorridorKeyToolboxPlugIn+FxAnalyzer.swift",
                "Parameters/CorridorKeyToolboxPlugIn+Parameters.swift",
                "Parameters/ParameterIdentifiers.swift",
                "Inference/InferenceCoordinator.swift",
                "Inference/KeyingInferenceEngine.swift",
                "Inference/MLXKeyingEngine.swift",
                "Inference/RoughMatteKeyingEngine.swift",
                "Inference/SharedMLXBridgeRegistry.swift",
                "Inspector/CorridorKeyToolboxPlugIn+CustomViews.swift",
                "Inspector/CorridorKeyInspectorBridge.swift",
                "Inspector/CorridorKeyHeaderView.swift",
                "Render",
                "Metal",
                "Resources",
                "Shared"
            ],
            sources: [
                "Parameters/ParameterEnumerations.swift",
                "Parameters/PluginStateData.swift",
                "PostProcess/ScreenColorEstimator.swift",
                "PostProcess/ColorGamutMatrix.swift",
                "PostProcess/TemporalBlender.swift",
                "Inference/AnalysisData.swift",
                "Inference/MatteCodec.swift",
                "Inspector/CorridorKeyAnalysisSnapshot.swift",
                "Inspector/WarmupStatus.swift"
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .target(
            name: "CorridorKeyToolboxMetalStages",
            dependencies: ["CorridorKeyToolboxLogic"],
            path: "CorridorKeyToolbox/Plugin",
            // This target owns Metal pool + MPS refiner + stage helpers. It
            // excludes every file that depends on FxPlug headers so it can
            // build from plain Swift + Metal + MetalPerformanceShaders.
            exclude: [
                "Info.plist",
                "CorridorKeyToolbox.entitlements",
                "PluginLog.swift",
                "main.swift",
                "CorridorKeyToolboxPlugIn.swift",
                "CorridorKeyToolboxPlugIn+Properties.swift",
                "CorridorKeyToolboxPlugIn+Render.swift",
                "CorridorKeyToolboxPlugIn+RenderRects.swift",
                "CorridorKeyToolboxPlugIn+PluginState.swift",
                "CorridorKeyToolboxPlugIn+FxAnalyzer.swift",
                "Parameters",
                "Inference",
                "Inspector",
                "PostProcess",
                "Resources",
                "Metal/FxImageTilePixelFormat.swift",
                "Render/RenderPipeline.swift",
                "Shared/CorridorKeyToolbox-Bridging-Header.h"
            ],
            sources: [
                "Shared/CorridorKeyShaderTypes.swift",
                "Metal/MetalDeviceCache.swift",
                "Metal/IntermediateTexturePool.swift",
                "Metal/MatteRefiner.swift",
                "Render/RenderStages.swift"
            ],
            resources: [
                .copy("Metal/CorridorKeyShaders.metal"),
                .copy("Shared/CorridorKeyShaderTypes.h")
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("CORRIDOR_KEY_SPM_MIRROR")
            ]
        ),
        .executableTarget(
            name: "BenchmarkRunner",
            dependencies: ["CorridorKeyToolboxLogic"],
            path: "BenchmarkRunner",
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .testTarget(
            name: "BenchmarkRunnerTests",
            dependencies: ["BenchmarkRunner"],
            path: "Tests/BenchmarkRunnerTests",
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .testTarget(
            name: "CorridorKeyToolboxLogicTests",
            dependencies: ["CorridorKeyToolboxLogic"],
            path: "Tests/CorridorKeyToolboxLogicTests",
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .testTarget(
            name: "CorridorKeyToolboxMetalStagesTests",
            dependencies: [
                "CorridorKeyToolboxMetalStages",
                "CorridorKeyToolboxLogic"
            ],
            path: "Tests/CorridorKeyToolboxMetalStagesTests",
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        )
    ]
)
