//
//  CorridorKeyShaders.metal
//  Corridor Key Toolbox
//
//  Metal kernels and vertex/fragment shaders that implement the Corridor Key
//  per-frame GPU pipeline: screen-colour rotation, downsample & normalisation
//  for neural inference, despill, alpha edge work (levels, gamma, erode/dilate,
//  blur), source passthrough, advanced refinement (refiner-strength blend,
//  light wrap, edge decontamination, connected-components despeckle), and
//  final compositing. The final compose is a render stage so the output goes
//  straight to Final Cut Pro's destination texture (which is a render target,
//  not a shader-writable texture).
//

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

#include "CorridorKeyShaderTypes.h"

// MARK: - Compose vertex/fragment (writes final RGBA to the FxPlug destination)

struct ComposeRasterizerData {
    float4 clipSpacePosition [[position]];
    float2 textureCoordinate;
};

/// Full-screen triangle strip positioned in pixel space. The viewport matches
/// the destination tile's pixel dimensions so clip-space positions derived by
/// `pixelPosition / viewport * 2` land correctly. Texture coordinates use
/// Metal's top-left origin convention (y=0 at top).
vertex ComposeRasterizerData corridorKeyComposeVertex(
    uint vertexID [[vertex_id]],
    constant CKVertex2D *vertices [[buffer(CKVertexInputIndexVertices)]],
    constant vector_uint2 *viewportSize [[buffer(CKVertexInputIndexViewportSize)]]
) {
    ComposeRasterizerData out;
    float2 pixelPosition = vertices[vertexID].position;
    float2 viewport = float2(*viewportSize);
    out.clipSpacePosition.xy = pixelPosition / (viewport * 0.5);
    out.clipSpacePosition.z = 0.0;
    out.clipSpacePosition.w = 1.0;
    out.textureCoordinate = vertices[vertexID].textureCoordinate;
    return out;
}

/// Reads source, foreground, and matte textures, applies the selected output
/// mode, and writes the final pixel into the destination colour attachment.
/// Runs at the destination resolution so there's no rescaling cost here.
///
/// The foreground texture is the refined network output after despill,
/// source passthrough, and inverse screen-colour rotation — it is in straight
/// RGB (0..1), same space as the source. "Processed" modes composite it;
/// "Source + Matte" keeps the raw user pixels for manual grading workflows.
fragment float4 corridorKeyComposeFragment(
    ComposeRasterizerData in [[stage_in]],
    texture2d<float, access::sample> sourceTexture     [[texture(CKTextureIndexSource)]],
    texture2d<float, access::sample> foregroundTexture [[texture(CKTextureIndexForeground)]],
    texture2d<float, access::sample> matteTexture      [[texture(CKTextureIndexMatte)]],
    constant CKComposeParams &params                   [[buffer(CKBufferIndexComposeParams)]]
) {
    constexpr sampler bilinear(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    float3 source     = sourceTexture.sample(bilinear, in.textureCoordinate).rgb;
    float3 foreground = foregroundTexture.sample(bilinear, in.textureCoordinate).rgb;
    float  alpha      = saturate(matteTexture.sample(bilinear, in.textureCoordinate).r);

    switch (params.outputMode) {
        case CKOutputModeMatteOnly:
            return float4(alpha, alpha, alpha, 1.0);
        case CKOutputModeForegroundOnly:
            return float4(foreground, 1.0);
        case CKOutputModeSourcePlusMatte:
            return float4(source * alpha, alpha);
        case CKOutputModeForegroundPlusMatte:
            return float4(foreground, alpha);
        case CKOutputModeProcessed:
        default:
            return float4(foreground * alpha, alpha);
    }
}

// MARK: - Screen colour domain mapping (compute-based)

/// Applies a 3x3 colour matrix to the source texture. Used to map blue-screen
/// content into the green domain (and back again).
kernel void corridorKeyApplyScreenMatrixKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant float3x3 &matrix [[buffer(CKBufferIndexScreenColorMatrix)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    float4 sample = source.read(gid);
    float3 rgb = matrix * sample.rgb;
    destination.write(float4(rgb, sample.a), gid);
}

// MARK: - Zero-copy MLX I/O

/// Normalises the source+hint into the NHWC-packed float32 tensor MLX
/// expects and writes it into a `device float*` buffer instead of a
/// texture. The render pipeline then hands `buffer.contents()` to
/// `MLXArray(rawPointer:)` so the inference graph reads this memory
/// directly — no CPU `[Float]` copy in between.
///
/// Layout: row-major `[1, rung, rung, 4]` floats; channels packed as
/// `(normR, normG, normB, hint)` per pixel.
kernel void corridorKeyNormalizeToBufferKernel(
    texture2d<float, access::sample> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::sample> hint [[texture(CKTextureIndexHint)]],
    device float *output [[buffer(0)]],
    constant CKNormalizeParams &params [[buffer(CKBufferIndexNormalizeParams)]],
    constant uint2 &dims [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    constexpr sampler areaSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(dims);
    float4 rgba = source.sample(areaSampler, uv);
    float hintValue = hint.sample(areaSampler, uv).r;
    float3 rec709 = params.workingToRec709 * rgba.rgb;
    float3 normalized = (rec709 - params.mean) * params.invStdDev;
    uint pixelIndex = gid.y * dims.x + gid.x;
    uint baseOffset = pixelIndex * 4u;
    output[baseOffset + 0u] = normalized.x;
    output[baseOffset + 1u] = normalized.y;
    output[baseOffset + 2u] = normalized.z;
    output[baseOffset + 3u] = hintValue;
}

/// Reads MLX's 1-channel alpha output buffer (layout `[1, H, W, 1]`) and
/// writes it into an `r32Float` texture. Flips y so the y-up bridge
/// layout matches the y-down texture convention the compose pass
/// samples with — identical to what `uploadCachedAlpha` does on CPU for
/// the analysed-cache path.
kernel void corridorKeyAlphaBufferToTextureKernel(
    device const float *input [[buffer(0)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    uint srcY = dims.y - 1u - gid.y;
    uint pixelIndex = srcY * dims.x + gid.x;
    float alpha = input[pixelIndex];
    destination.write(float4(alpha, 0.0, 0.0, 1.0), gid);
}

/// Reads MLX's 3-channel foreground output buffer (layout `[1, H, W, 3]`)
/// and writes it into an `rgba32Float` texture with alpha = 1. Flips y
/// for the same reason as the alpha kernel. Replaces the old CPU-side
/// `cblas_scopy` RGB→RGBA interleave — now the channel expansion
/// happens on the GPU in the same pass as the buffer read.
kernel void corridorKeyForegroundBufferToTextureKernel(
    device const float *input [[buffer(0)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    uint srcY = dims.y - 1u - gid.y;
    uint pixelIndex = srcY * dims.x + gid.x;
    uint baseOffset = pixelIndex * 3u;
    float3 rgb = float3(input[baseOffset], input[baseOffset + 1u], input[baseOffset + 2u]);
    destination.write(float4(rgb, 1.0), gid);
}

// MARK: - Normalisation (texture output — used only by golden tests)

/// Downsamples source and hint to the inference resolution and produces the
/// four-channel tensor the neural model expects (RGB mean/stddev normalised,
/// hint packed into alpha). The destination texture's dimensions set the
/// inference resolution; bilinear filtering handles the rescale. The
/// `workingToRec709` matrix converts whatever working colour space the host
/// provided into the Rec.709-linear sRGB the model was trained on.
///
/// Production renders use `corridorKeyNormalizeToBufferKernel` above for
/// zero-copy interop with MLX; this texture-output variant stays for
/// test fixtures that want to read the normalised tensor back as a
/// conventional `.rgba32Float` texture.
kernel void corridorKeyCombineAndNormalizeKernel(
    texture2d<float, access::sample> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::sample> hint [[texture(CKTextureIndexHint)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKNormalizeParams &params [[buffer(CKBufferIndexNormalizeParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    constexpr sampler areaSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    float2 uv = (float2(gid) + 0.5) / float2(dims);
    float4 rgba = source.sample(areaSampler, uv);
    float hintValue = hint.sample(areaSampler, uv).r;

    float3 rec709 = params.workingToRec709 * rgba.rgb;
    float3 normalized = (rec709 - params.mean) * params.invStdDev;
    destination.write(float4(normalized, hintValue), gid);
}

// MARK: - Despill

/// Corridor Key's despill runs in linear RGB. Green is assumed to be the screen
/// colour; callers rotate blue-screen content into the green domain first.
kernel void corridorKeyDespillKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKDespillParams &params [[buffer(CKBufferIndexDespillParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = source.read(gid);
    float r = rgba.r;
    float g = rgba.g;
    float b = rgba.b;

    float limit = 0.0;
    if (params.method == CKSpillMethodDoubleLimit) {
        limit = max(r, b);
    } else {
        limit = (r + b) * 0.5;
    }

    float spill = max(0.0, g - limit);
    if (spill > 0.0 && params.strength > 0.0) {
        float effectiveSpill = spill * params.strength;
        float newG = g - effectiveSpill;

        if (params.method == CKSpillMethodNeutral) {
            // Guard against division by near-zero on dark pixels. The
            // reference CorridorKey-Runtime uses 1e-3 (src/post_process/
            // despill.cpp) to avoid the noise amplification a 1e-6 floor
            // produced on dark skin tones / hair.
            float gray = (r + newG + b) * (1.0 / 3.0);
            float fill = effectiveSpill * 0.5;
            r = r + fill * (gray / max(r, 1e-3));
            b = b + fill * (gray / max(b, 1e-3));
            // Clamp the result to [0, 1] so extreme mixers can't blow out
            // the foreground — matches the reference behaviour.
            r = saturate(r);
            b = saturate(b);
        } else {
            r = r + effectiveSpill * 0.5;
            b = b + effectiveSpill * 0.5;
        }
        g = newG;
    }

    destination.write(float4(r, g, b, rgba.a), gid);
}

// MARK: - Alpha levels + gamma

kernel void corridorKeyAlphaLevelsGammaKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKAlphaEdgeParams &params [[buffer(CKBufferIndexAlphaEdgeParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float alpha = source.read(gid).r;
    float range = max(params.whitePoint - params.blackPoint, 1e-6);
    alpha = saturate((alpha - params.blackPoint) / range);
    if (params.gamma > 0.0 && params.gamma != 1.0 && alpha > 0.0 && alpha < 1.0) {
        alpha = pow(alpha, 1.0 / params.gamma);
    }
    destination.write(float4(alpha, 0.0, 0.0, 1.0), gid);
}

// MARK: - Morphology (separable erode / dilate — fallback for tiny radii)

kernel void corridorKeyMorphologyHorizontalKernel(
    texture2d<float, access::sample> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant int &radius [[buffer(0)]],
    constant int &erode [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    constexpr sampler clampSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 invSize = 1.0 / float2(dims);
    float2 uv = (float2(gid) + 0.5) * invSize;

    float best = source.sample(clampSampler, uv).r;
    for (int dx = 1; dx <= radius; ++dx) {
        float left = source.sample(clampSampler, uv + float2(-dx * invSize.x, 0.0)).r;
        float right = source.sample(clampSampler, uv + float2(dx * invSize.x, 0.0)).r;
        if (erode != 0) {
            best = min(best, min(left, right));
        } else {
            best = max(best, max(left, right));
        }
    }
    destination.write(float4(best, 0.0, 0.0, 1.0), gid);
}

kernel void corridorKeyMorphologyVerticalKernel(
    texture2d<float, access::sample> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant int &radius [[buffer(0)]],
    constant int &erode [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    constexpr sampler clampSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 invSize = 1.0 / float2(dims);
    float2 uv = (float2(gid) + 0.5) * invSize;

    float best = source.sample(clampSampler, uv).r;
    for (int dy = 1; dy <= radius; ++dy) {
        float up = source.sample(clampSampler, uv + float2(0.0, -dy * invSize.y)).r;
        float down = source.sample(clampSampler, uv + float2(0.0, dy * invSize.y)).r;
        if (erode != 0) {
            best = min(best, min(up, down));
        } else {
            best = max(best, max(up, down));
        }
    }
    destination.write(float4(best, 0.0, 0.0, 1.0), gid);
}

// MARK: - Gaussian blur (separable — fallback for tiny radii)

kernel void corridorKeyGaussianHorizontalKernel(
    texture2d<float, access::sample> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant float *weights [[buffer(CKBufferIndexBlurWeights)]],
    constant int &kernelRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    constexpr sampler clampSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 invSize = 1.0 / float2(dims);
    float2 uv = (float2(gid) + 0.5) * invSize;

    float acc = source.sample(clampSampler, uv).r * weights[0];
    for (int i = 1; i <= kernelRadius; ++i) {
        float w = weights[i];
        float left = source.sample(clampSampler, uv + float2(-i * invSize.x, 0.0)).r;
        float right = source.sample(clampSampler, uv + float2(i * invSize.x, 0.0)).r;
        acc += (left + right) * w;
    }
    destination.write(float4(acc, 0.0, 0.0, 1.0), gid);
}

kernel void corridorKeyGaussianVerticalKernel(
    texture2d<float, access::sample> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant float *weights [[buffer(CKBufferIndexBlurWeights)]],
    constant int &kernelRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    constexpr sampler clampSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 invSize = 1.0 / float2(dims);
    float2 uv = (float2(gid) + 0.5) * invSize;

    float acc = source.sample(clampSampler, uv).r * weights[0];
    for (int i = 1; i <= kernelRadius; ++i) {
        float w = weights[i];
        float up = source.sample(clampSampler, uv + float2(0.0, -i * invSize.y)).r;
        float down = source.sample(clampSampler, uv + float2(0.0, i * invSize.y)).r;
        acc += (up + down) * w;
    }
    destination.write(float4(acc, 0.0, 0.0, 1.0), gid);
}

// MARK: - Green screen detection / rough matte fallback

/// Coarse alpha-hint for the neural model. Matches the CorridorKey-Runtime
/// reference (`ColorUtils::generate_rough_matte` in
/// src/post_process/color_utils.cpp): the hint uses matte convention —
/// `1.0` means foreground/keep, `0.0` means screen/remove — so the trained
/// network reads it the same way it was trained. Sending raw greenness
/// here flips the model's output and the whole key inverts. This single
/// kernel replaces the previous separate `roughMatte` / `greenHint` kernels
/// which were byte-identical.
kernel void corridorKeyGreenHintKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = source.read(gid);
    float greenBias = rgba.g - max(rgba.r, rgba.b);
    float matte = 1.0 - saturate(greenBias * 2.0);
    destination.write(float4(matte, 0.0, 0.0, 1.0), gid);
}

// MARK: - Source passthrough blending

kernel void corridorKeySourcePassthroughKernel(
    texture2d<float, access::read> foreground [[texture(CKTextureIndexForeground)]],
    texture2d<float, access::read> sourceRGB [[texture(CKTextureIndexSource)]],
    texture2d<float, access::read> mask [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float m = saturate(mask.read(gid).r);
    float3 fg = foreground.read(gid).rgb;
    if (m <= 0.0) {
        destination.write(float4(fg, 1.0), gid);
        return;
    }
    float3 src = sourceRGB.read(gid).rgb;
    float3 blended = m * src + (1.0 - m) * fg;
    destination.write(float4(blended, 1.0), gid);
}

// MARK: - Resample (bilinear fallback; MPS Lanczos handles Quality = Lanczos)

kernel void corridorKeyResampleKernel(
    texture2d<float, access::sample> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    constexpr sampler bilinear(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(dims);
    destination.write(source.sample(bilinear, uv), gid);
}

// MARK: - Hint ingestion

kernel void corridorKeyExtractHintKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant int &sourceLayout [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = source.read(gid);
    float hint = 0.0;
    // 0 = RGBA → use alpha, 1 = alpha only / R only, 2 = RGB → use red.
    if (sourceLayout == 0) {
        hint = rgba.a;
    } else {
        hint = rgba.r;
    }
    destination.write(float4(hint, 0.0, 0.0, 1.0), gid);
}

// MARK: - Fused matte refine (levels + gamma + refiner blend)

/// Collapses three per-pixel matte passes into one dispatch. The old
/// path ran `alphaLevelsGamma` → `refinerBlend` as two separate compute
/// encoders, each reading and writing a full matte texture. This kernel
/// does both stages in-register, cutting ~1 ms per 4K render and one
/// texture read/write of bandwidth.
kernel void corridorKeyMatteRefineKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::read> coarse [[texture(CKTextureIndexCoarse)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKMatteRefineParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float alpha = source.read(gid).r;

    // Levels + gamma (identical to `corridorKeyAlphaLevelsGammaKernel`).
    float range = max(params.whitePoint - params.blackPoint, 1e-6);
    alpha = saturate((alpha - params.blackPoint) / range);
    if (params.gamma > 0.0 && params.gamma != 1.0 && alpha > 0.0 && alpha < 1.0) {
        alpha = pow(alpha, 1.0 / params.gamma);
    }

    // Refiner strength blend. When strength is a no-op (1.0) skip the
    // coarse-texture read and write `alpha` through directly — saves
    // ~15 ms of effectively-wasted bandwidth on every 4K frame where
    // the user hasn't moved the refiner slider.
    if (abs(params.refinerStrength - 1.0) > 1e-3) {
        float coarseAlpha = coarse.read(gid).r;
        float blended = coarseAlpha + (alpha - coarseAlpha) * params.refinerStrength;
        alpha = saturate(blended);
    }

    destination.write(float4(alpha, 0.0, 0.0, 1.0), gid);
}

// MARK: - Fused foreground post-process (passthrough + light-wrap + decontam + inverse-rotation)

/// Collapses four per-pixel foreground passes into a single dispatch.
/// The old path ran
///
///   sourcePassthrough → (lightWrap) → (edgeDecontam) → applyScreenMatrix
///
/// each as its own compute encoder. This kernel folds them into one
/// in-register sequence so the intermediate RGB never needs to make a
/// texture → texture bandwidth round-trip. On 4K that saves about 3 ms
/// per frame (3 texture writes + 3 texture reads at rgba32Float ≈ 384
/// MB of bandwidth) plus three command-encoder dispatch overheads.
///
/// Disabled stages are guarded by the `*Enabled` flags in the params
/// struct; the GPU branches coherently across the whole grid so
/// turned-off stages cost essentially nothing.
kernel void corridorKeyForegroundPostProcessKernel(
    texture2d<float, access::read> foreground [[texture(CKTextureIndexForeground)]],
    texture2d<float, access::read> sourceRGB [[texture(CKTextureIndexSource)]],
    texture2d<float, access::read> matte [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::read> blurredSource [[texture(CKTextureIndexHint)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKForegroundPostProcessParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float3 rgb = foreground.read(gid).rgb;
    float  m   = saturate(matte.read(gid).r);

    // Stage 1: source passthrough. Matches the standalone
    // `corridorKeySourcePassthroughKernel` — when the matte says
    // "full foreground" the user's original source pixels show through
    // instead of the network-reconstructed foreground.
    if (params.sourcePassthroughEnabled != 0 && m > 0.0) {
        float3 src = sourceRGB.read(gid).rgb;
        rgb = m * src + (1.0 - m) * rgb;
    }

    // Stage 2: light wrap. Blurred-source ambient additively blends
    // into the foreground along the soft edge.
    if (params.lightWrapEnabled != 0 && params.lightWrapStrength > 0.0) {
        float3 wrap = blurredSource.read(gid).rgb;
        float falloff = 1.0 - m;
        float biased = pow(falloff, mix(1.0, 4.0, saturate(params.lightWrapEdgeBias)));
        rgb = rgb + wrap * biased * params.lightWrapStrength * m;
    }

    // Stage 3: edge colour decontamination.
    if (params.edgeDecontaminateEnabled != 0 && params.edgeDecontaminateStrength > 0.0) {
        float3 screen = params.screenColor;
        float screenLen = max(length(screen), 1e-3);
        float3 screenDir = screen / screenLen;
        float residual = max(dot(rgb, screenDir), 0.0);
        rgb = rgb - screenDir * residual * (1.0 - m) * params.edgeDecontaminateStrength;
        rgb = max(rgb, float3(0.0));
    }

    // Stage 4: inverse screen-colour rotation. Identity bypass for
    // green screens; the matrix multiply here costs 9 mul-adds when
    // applied so we gate it on a flag.
    if (params.applyInverseRotation != 0) {
        rgb = params.inverseScreenMatrix * rgb;
    }

    destination.write(float4(rgb, 1.0), gid);
}

// MARK: - Refiner-strength blend (Phase 4.1)

/// Blends the model's refined matte with a pre-blurred "coarse" stand-in
/// produced by a Gaussian blur. `strength == 1.0` is a no-op (refined pass
/// through). `strength < 1.0` biases toward the blurred stand-in (softer
/// edges). `strength > 1.0` extrapolates toward sharper edges, clamped to
/// [0, 1]. See the plan's phase 4.1 note on why this is a post-hoc
/// approximation for the model's built-in refiner scale.
kernel void corridorKeyRefinerBlendKernel(
    texture2d<float, access::read> refined [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::read> coarse [[texture(CKTextureIndexCoarse)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKRefinerParams &params [[buffer(CKBufferIndexRefinerParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float refinedAlpha = refined.read(gid).r;
    float coarseAlpha = coarse.read(gid).r;
    float blended = coarseAlpha + (refinedAlpha - coarseAlpha) * params.strength;
    destination.write(float4(saturate(blended), 0.0, 0.0, 1.0), gid);
}

// MARK: - Temporal blend (Phase 1 — matte flicker reduction)

/// Blends the current frame's alpha toward the previous frame's alpha on
/// pixels where the input RGB barely changed — the classic symptom of
/// per-frame neural inference noise. Runs at inference resolution during
/// the analysis pass so playback renders inherit stabilised mattes with
/// zero hot-path cost.
///
/// `motionThreshold` is the max-channel absolute RGB delta at which the
/// gate reaches the first zero-blend step. Values above the threshold
/// scale linearly down to zero at `2 × motionThreshold`, and any larger
/// delta passes the current alpha through unchanged so fast action
/// retains full per-frame fidelity.
///
/// The previous-frame textures are produced by the matching compute
/// encoder on the previous analysis tick; `CorridorKeyToolboxPlugIn+FxAnalyzer`
/// keeps them alive across ticks via a ring of two intermediate textures.
kernel void corridorKeyTemporalBlendKernel(
    texture2d<float, access::read> currentMatte     [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::read> previousMatte    [[texture(CKTextureIndexPreviousMatte)]],
    texture2d<float, access::read> currentSource    [[texture(CKTextureIndexSource)]],
    texture2d<float, access::read> previousSource   [[texture(CKTextureIndexPreviousSource)]],
    texture2d<float, access::write> destination     [[texture(CKTextureIndexOutput)]],
    constant CKTemporalBlendParams &params          [[buffer(CKBufferIndexTemporalBlendParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float currentAlpha = currentMatte.read(gid).r;
    float previousAlpha = previousMatte.read(gid).r;

    float3 currentRGB = currentSource.read(gid).rgb;
    float3 previousRGB = previousSource.read(gid).rgb;
    float3 deltaRGB = abs(currentRGB - previousRGB);
    float motion = max(deltaRGB.x, max(deltaRGB.y, deltaRGB.z));

    // Linear falloff: gate = 1 at motion == 0, gate = 0 at motion == 2 × threshold.
    // Guard against a zero threshold — callers disable temporal blending
    // entirely by setting `strength == 0`, but defensive clamping keeps
    // the shader robust if a future code path forgets the guard.
    float threshold = max(params.motionThreshold, 1e-6);
    float motionGate = saturate(1.0 - motion / (2.0 * threshold));
    float effectiveStrength = saturate(params.strength) * motionGate;

    float blended = currentAlpha + (previousAlpha - currentAlpha) * effectiveStrength;
    destination.write(float4(saturate(blended), 0.0, 0.0, 1.0), gid);
}

// MARK: - Light wrap (Phase 4.3)

/// Simulates environment lighting wrapping onto the subject near matte
/// edges. `sourceBlur` is a pre-blurred copy of the original source RGB; the
/// kernel reads that outside-the-matte colour and additively blends it into
/// the foreground along the falloff `(1 - matte)`.
kernel void corridorKeyLightWrapKernel(
    texture2d<float, access::read> foreground [[texture(CKTextureIndexForeground)]],
    texture2d<float, access::read> sourceBlur [[texture(CKTextureIndexSource)]],
    texture2d<float, access::read> matte [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKLightWrapParams &params [[buffer(CKBufferIndexLightWrapParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float3 fg = foreground.read(gid).rgb;
    float m = saturate(matte.read(gid).r);
    // A thin ring near the matte edge gets the strongest wrap; interiors
    // (`m` near 1) are untouched. `edgeBias` biases the falloff curve.
    float falloff = 1.0 - m;
    float biased = pow(falloff, mix(1.0, 4.0, saturate(params.edgeBias)));
    float3 wrap = sourceBlur.read(gid).rgb;
    float3 blended = fg + wrap * biased * params.strength * m;
    destination.write(float4(blended, 1.0), gid);
}

// MARK: - Edge colour decontamination (Phase 4.4)

/// Removes residual screen colour from foreground RGB in the matte edge
/// band (where `0 < matte < 1`). Applies a per-pixel screen-colour subtract
/// weighted by `(1 - matte) * strength`. The opaque interior stays
/// untouched, and the fully transparent regions don't contribute to the
/// composite anyway.
kernel void corridorKeyEdgeDecontaminateKernel(
    texture2d<float, access::read> foreground [[texture(CKTextureIndexForeground)]],
    texture2d<float, access::read> matte [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKEdgeDecontaminateParams &params [[buffer(CKBufferIndexEdgeDecontaminateParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float3 fg = foreground.read(gid).rgb;
    float m = saturate(matte.read(gid).r);
    // Weight by (1 - m) so the interior is untouched. The subtraction
    // component is the "how much screen colour is still in this pixel" —
    // we project fg onto the screen colour direction and scale.
    float3 screen = params.screenColor;
    float screenLen = max(length(screen), 1e-3);
    float3 screenDir = screen / screenLen;
    float residual = max(dot(fg, screenDir), 0.0);
    float3 decont = fg - screenDir * residual * (1.0 - m) * params.strength;
    destination.write(float4(max(decont, float3(0.0)), 1.0), gid);
}

// MARK: - Connected-components despeckle (Phase 4.2)

/// Initialises the label texture from a binarised matte. Every foreground
/// pixel receives a unique integer ID packed into the R32 channel; every
/// background pixel receives `0`. The subsequent propagate pass flood-fills
/// labels to the minimum ID inside each connected component.
kernel void corridorKeyCCLabelInitKernel(
    texture2d<float, access::read> matte [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> labelOut [[texture(CKTextureIndexOutput)]],
    constant CKCCLabelParams &params [[buffer(CKBufferIndexCCLabelParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(labelOut.get_width(), labelOut.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float alpha = matte.read(gid).r;
    if (alpha < params.matteThreshold) {
        labelOut.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    // Labels are stored as linear (y * width + x) + 1, so 0 stays reserved
    // for "background". We keep them in a float32 texture (r32Float) —
    // labels up to ~16M are exactly representable, enough for a 4096² matte.
    float label = float(gid.y * dims.x + gid.x + 1u);
    labelOut.write(float4(label, 0.0, 0.0, 1.0), gid);
}

/// One iteration of label propagation: each pixel takes the minimum label
/// of itself and its 8 neighbours sampled at `stride` texels away. Caller
/// drives `stride = 1, 2, 4, 8, …` so the propagation "pointer-jumps" in
/// `log₂(max(W, H))` iterations instead of `max(W, H)` — crucial for 4K
/// mattes where the 1-pixel-stride version would need thousands of passes.
kernel void corridorKeyCCLabelPropagateKernel(
    texture2d<float, access::read> labelIn [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> labelOut [[texture(CKTextureIndexOutput)]],
    constant int &stride [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(labelOut.get_width(), labelOut.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float current = labelIn.read(gid).r;
    if (current <= 0.0) {
        labelOut.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    float best = current;
    int s = max(stride, 1);
    const int2 offsets[8] = {
        int2(-s, -s), int2(0, -s), int2(s, -s),
        int2(-s, 0),               int2(s, 0),
        int2(-s, s),  int2(0, s),  int2(s, s)
    };
    for (int i = 0; i < 8; ++i) {
        int2 neighbour = int2(gid) + offsets[i];
        if (neighbour.x < 0 || neighbour.y < 0 ||
            neighbour.x >= int(dims.x) || neighbour.y >= int(dims.y)) { continue; }
        float value = labelIn.read(uint2(neighbour)).r;
        if (value > 0.0 && value < best) {
            best = value;
        }
    }
    labelOut.write(float4(best, 0.0, 0.0, 1.0), gid);
}

/// Doubling pointer-jump pass. After stride-1 propagation has linked
/// each pixel to a lower-label "parent" within its local neighbourhood,
/// this kernel classic-union-find-compresses the resulting chains:
/// each pixel replaces its label with the label at the *pointed-to*
/// position. Repeated `log₂(chain length)` times, every pixel ends up
/// labelled with its component's global minimum. This is far cheaper
/// than another `N` stride-1 iterations on a 4K matte.
kernel void corridorKeyCCLabelPointerJumpKernel(
    texture2d<float, access::read> labelIn [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> labelOut [[texture(CKTextureIndexOutput)]],
    constant int &matteWidth [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(labelOut.get_width(), labelOut.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float current = labelIn.read(gid).r;
    if (current <= 0.0) {
        labelOut.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    uint currentLabel = uint(current);
    uint linearIndex = currentLabel - 1u;
    uint width = uint(max(matteWidth, 1));
    uint targetX = linearIndex % width;
    uint targetY = linearIndex / width;
    if (targetX >= dims.x || targetY >= dims.y) {
        labelOut.write(float4(current, 0.0, 0.0, 1.0), gid);
        return;
    }
    float pointed = labelIn.read(uint2(targetX, targetY)).r;
    if (pointed > 0.0 && pointed < current) {
        labelOut.write(float4(pointed, 0.0, 0.0, 1.0), gid);
    } else {
        labelOut.write(float4(current, 0.0, 0.0, 1.0), gid);
    }
}

/// Counts pixels per component into a shared atomic buffer. The filter
/// pass reads from the same buffer to zero components below threshold.
/// Using atomics lets the whole CC despeckle fit on a single command
/// buffer without any CPU readback.
kernel void corridorKeyCCLabelCountKernel(
    texture2d<float, access::read> labelIn [[texture(CKTextureIndexLabel)]],
    device atomic_uint *labelCounts [[buffer(CKBufferIndexCCLabelCounts)]],
    constant CKCCLabelParams &params [[buffer(CKBufferIndexCCLabelParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(labelIn.get_width(), labelIn.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float label = labelIn.read(gid).r;
    if (label <= 0.0) { return; }
    uint labelIndex = uint(label);
    // `labelSpan` carries the raw capacity of the counts buffer (width *
    // height + 1, to cover label values `1…width*height`). Swift sets it
    // this way so non-square mattes work.
    uint capacity = uint(params.labelSpan);
    if (labelIndex >= capacity) { return; }
    atomic_fetch_add_explicit(&labelCounts[labelIndex], 1u, memory_order_relaxed);
}

/// Applies the despeckle decision. Reads each pixel's (now stabilised)
/// label and consults the `labelCounts` buffer — if the component's area is
/// below `areaThreshold`, the pixel is zeroed; otherwise it passes
/// through the original matte alpha. `labelCapacity` bounds-checks the
/// counts buffer so a stray label value can't over-read. Both values live
/// in the `CKCCLabelParams` slot — the Swift side packs `areaThreshold`
/// and a capacity sentinel (`labelSpan * labelSpan`) into the struct.
kernel void corridorKeyCCLabelFilterKernel(
    texture2d<float, access::read> labelIn [[texture(CKTextureIndexLabel)]],
    texture2d<float, access::read> matte [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    device const uint *labelCounts [[buffer(CKBufferIndexCCLabelCounts)]],
    constant CKCCLabelParams &params [[buffer(CKBufferIndexCCLabelParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float alpha = matte.read(gid).r;
    float label = labelIn.read(gid).r;
    // Pixel wasn't binarised as foreground (below `matteThreshold` at init
    // time). These are soft-edge pixels — hair, transparent halos, the
    // despill tail — and we MUST preserve their original alpha. The
    // despeckle filter only targets confident-foreground specks that the
    // neural model hallucinated.
    if (label <= 0.0) {
        destination.write(float4(alpha, 0.0, 0.0, 1.0), gid);
        return;
    }
    uint labelIndex = uint(label);
    // `labelSpan` carries the raw capacity of the counts buffer (width *
    // height + 1, to cover label values `1…width*height`). Swift sets it
    // this way so non-square mattes work.
    uint capacity = uint(params.labelSpan);
    if (labelIndex >= capacity) {
        destination.write(float4(alpha, 0.0, 0.0, 1.0), gid);
        return;
    }
    uint area = labelCounts[labelIndex];
    if (int(area) < params.areaThreshold) {
        destination.write(float4(0.0, 0.0, 0.0, 1.0), gid);
    } else {
        destination.write(float4(alpha, 0.0, 0.0, 1.0), gid);
    }
}
