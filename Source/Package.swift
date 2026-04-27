// swift-tools-version: 6.0
//
//  Package.swift
//  CorridorKey by LateNite
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
    dependencies: [
        // Mirrors the same `mlx-swift` reference the Xcode project uses so
        // we can unit-test the MLX inference path outside Final Cut Pro.
        // Pinned to the same release the production target consumes.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.3")
    ],
    targets: [
        .target(
            name: "CorridorKeyToolboxLogic",
            path: "CorridorKeyByLateNite/Plugin",
            // Keep the FxPlug-dependent files, the Xcode-managed resources,
            // and the shaders out of the SPM build — they only compile inside
            // the Xcode target.
            exclude: [
                "Info.plist",
                "CorridorKeyByLateNite.entitlements",
                "PluginLog.swift",
                "main.swift",
                "CorridorKeyByLateNitePlugIn.swift",
                "CorridorKeyByLateNitePlugIn+Properties.swift",
                "CorridorKeyByLateNitePlugIn+Render.swift",
                "CorridorKeyByLateNitePlugIn+RenderRects.swift",
                "CorridorKeyByLateNitePlugIn+PluginState.swift",
                "CorridorKeyByLateNitePlugIn+FxAnalyzer.swift",
                "Parameters/CorridorKeyByLateNitePlugIn+Parameters.swift",
                "Parameters/ParameterIdentifiers.swift",
                "Inference/InferenceCoordinator.swift",
                "Inference/KeyingInferenceEngine.swift",
                "Inference/MLXKeyingEngine.swift",
                "Inference/SharedMLXBridgeRegistry.swift",
                "Inference/VisionHintEngine.swift",
                "Inspector/CorridorKeyByLateNitePlugIn+CustomViews.swift",
                "Inspector/CorridorKeyInspectorBridge.swift",
                "Inspector/CorridorKeyHeaderView.swift",
                "Inspector/CorridorKeyHintOSC.swift",
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
                "Inference/DeviceCapabilityCache.swift",
                "Inference/MatteCodec.swift",
                "Inspector/CorridorKeyAnalysisSnapshot.swift",
                "Inspector/HintPointSet.swift",
                "Inspector/WarmupStatus.swift"
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .target(
            name: "CorridorKeyToolboxMetalStages",
            dependencies: [
                "CorridorKeyToolboxLogic",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            path: "CorridorKeyByLateNite/Plugin",
            // This target owns the Metal pool + MPS refiner + stage helpers
            // **and** the inference engines (MLX + rough-matte fallback).
            // Both layers share `MetalDeviceCacheEntry` and `RenderStages`,
            // so collapsing them into one SPM module sidesteps having to
            // promote large swathes of internal API to `public`.
            // Excludes every file that depends on FxPlug headers so it can
            // build from plain Swift + Metal + MPS + mlx-swift.
            exclude: [
                "Info.plist",
                "CorridorKeyByLateNite.entitlements",
                "main.swift",
                "CorridorKeyByLateNitePlugIn.swift",
                "CorridorKeyByLateNitePlugIn+Properties.swift",
                "CorridorKeyByLateNitePlugIn+Render.swift",
                "CorridorKeyByLateNitePlugIn+RenderRects.swift",
                "CorridorKeyByLateNitePlugIn+PluginState.swift",
                "CorridorKeyByLateNitePlugIn+FxAnalyzer.swift",
                "Parameters",
                "Inspector",
                "PostProcess",
                "Resources",
                "Metal/FxImageTilePixelFormat.swift",
                "Render/RenderPipeline.swift",
                "Shared/CorridorKeyByLateNite-Bridging-Header.h",
                "Inference/AnalysisData.swift",
                "Inference/DeviceCapabilityCache.swift",
                "Inference/MatteCodec.swift"
            ],
            sources: [
                "PluginLog.swift",
                "Shared/CorridorKeyShaderTypes.swift",
                "Metal/MetalDeviceCache.swift",
                "Metal/IntermediateTexturePool.swift",
                "Metal/MatteRefiner.swift",
                "Metal/MetalStagesResourceBundleAccessor.swift",
                "Render/RenderStages.swift",
                "Inference/KeyingInferenceEngine.swift",
                "Inference/MLXKeyingEngine.swift",
                "Inference/InferenceCoordinator.swift",
                "Inference/SharedMLXBridgeRegistry.swift",
                "Inference/VisionHintEngine.swift"
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
            name: "CorridorKeyByLateNiteLogicTests",
            dependencies: ["CorridorKeyToolboxLogic"],
            path: "Tests/CorridorKeyByLateNiteLogicTests",
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .testTarget(
            name: "CorridorKeyByLateNiteMetalStagesTests",
            dependencies: [
                "CorridorKeyToolboxMetalStages",
                "CorridorKeyToolboxLogic"
            ],
            path: "Tests/CorridorKeyByLateNiteMetalStagesTests",
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .testTarget(
            name: "CorridorKeyByLateNiteInferenceTests",
            dependencies: [
                "CorridorKeyToolboxMetalStages",
                "CorridorKeyToolboxLogic",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Tests/CorridorKeyByLateNiteInferenceTests",
            // The 512px bridge is ~30 MB and exercises the same MLX code
            // path as the Maximum rung. Bundling just the smallest one
            // keeps the test target compact while still catching real
            // memory regressions.
            resources: [
                .copy("Resources/corridorkey_mlx_bridge_512.mlxfn")
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        )
    ]
)
