//
//  KeyingInferenceEngine.swift
//  Corridor Key Toolbox
//
//  Abstraction over whichever neural backend is producing mattes. The render
//  pipeline is written against this protocol so we can swap CoreML, MLX, or a
//  pure-GPU rough-matte fallback without touching the FxPlug integration.
//

import Foundation
import Metal
import CoreMedia

/// Input texture bundle handed to an inference engine. The pipeline provides
/// both the ImageNet-normalised tensor (for MLX / CoreML) and the raw RGB
/// source at the same inference resolution so GPU-only fallbacks can compute
/// their matte directly without re-parsing the normalised one.
struct KeyingInferenceRequest {
    /// RGBA (hint packed into alpha) tensor normalised with ImageNet stats,
    /// sized `inferenceResolution × inferenceResolution`. `.shared` storage
    /// so CPU-backed engines (MLX) can read it back.
    let normalisedInputTexture: any MTLTexture
    /// Raw RGB source at inference resolution, in 0..1 linear/sRGB space.
    /// Any GPU engine that doesn't need the neural network should read from
    /// this texture instead of the normalised one.
    let rawSourceTexture: any MTLTexture
    /// Square side length used for inference.
    let inferenceResolution: Int
}

/// Textures the engine writes into. Both are created by the caller at the
/// requested inference resolution.
struct KeyingInferenceOutput {
    let alphaTexture: any MTLTexture
    let foregroundTexture: any MTLTexture
}

/// Error surface returned when inference cannot proceed.
enum KeyingInferenceError: Error, CustomStringConvertible {
    case modelUnavailable(String)
    case unsupportedResolution(Int)
    case deviceUnavailable

    var description: String {
        switch self {
        case .modelUnavailable(let detail):
            return "Keying model is unavailable: \(detail)"
        case .unsupportedResolution(let resolution):
            return "This build does not include a model tuned for \(resolution)px inference."
        case .deviceUnavailable:
            return "A Metal device capable of running the Corridor Key model was not found."
        }
    }
}

/// Minimal surface the orchestrator exercises. Concrete engines own their
/// session lifetime (model load, warm-up) but must be thread-safe because
/// Final Cut Pro may render multiple frames simultaneously.
protocol KeyingInferenceEngine: AnyObject, Sendable {
    var backendDisplayName: String { get }
    var guideSourceDescription: String { get set }
    func supports(resolution: Int) -> Bool
    func prepare(resolution: Int) async throws
    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws
}
