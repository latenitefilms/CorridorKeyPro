//
//  KeyingInferenceEngine.swift
//  Corridor Key Pro
//
//  Abstraction over whichever neural backend is producing mattes. The render
//  pipeline is written against this protocol so we can swap CoreML, MLX, or a
//  pure-GPU rough-matte fallback without touching the FxPlug integration.
//

import Foundation
import Metal
import CoreMedia

/// The shape of the tensors the engine consumes and produces. All engines must
/// be able to handle at least one resolution from the Corridor Key quality
/// ladder (512, 1024, 1536, 2048 on the long edge).
///
/// By the time `KeyingInferenceEngine.run(request:output:)` is invoked the
/// render pipeline has already committed and awaited the pre-inference GPU
/// pass, so `normalisedInputTexture` holds valid contents and is allocated
/// with `.shared` storage for CPU readback. Engines must not enqueue work
/// onto the render pipeline's command buffer — they either do their own
/// GPU work on a fresh command buffer or execute the computation on the
/// CPU (as `MLXKeyingEngine` does when consulting MLX).
struct KeyingInferenceRequest {
    /// RGBA (hint packed into alpha) texture, already normalised and resampled
    /// to `inferenceResolution x inferenceResolution` by the render pipeline.
    /// Allocated with `.shared` storage.
    let normalisedInputTexture: MTLTexture
    /// Square side length used for inference.
    let inferenceResolution: Int
}

/// Textures the engine writes into. Both are created by the caller at the
/// requested inference resolution.
struct KeyingInferenceOutput {
    let alphaTexture: MTLTexture
    let foregroundTexture: MTLTexture
}

/// Error surface returned when inference cannot proceed. The orchestrator can
/// downgrade the quality rung or fall back to a rough matte depending on which
/// case is returned.
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
    /// Human-readable label surfaced in the status parameter.
    var backendDisplayName: String { get }
    /// Guide source label displayed next to the matte.
    var guideSourceDescription: String { get set }
    /// Whether this engine can service a request at the given resolution. The
    /// orchestrator uses this to pick a fallback rung when needed.
    func supports(resolution: Int) -> Bool
    /// Ensures the model is loaded and warmed up for the requested resolution.
    /// Must be idempotent.
    func prepare(resolution: Int) async throws
    /// Runs inference on the supplied tensor. Engines enqueue their work onto
    /// the request's command buffer rather than committing themselves.
    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws
}
