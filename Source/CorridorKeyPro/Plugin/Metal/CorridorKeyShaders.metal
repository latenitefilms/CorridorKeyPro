//
//  CorridorKeyShaders.metal
//  Corridor Key Toolbox
//
//  Metal kernels and vertex/fragment shaders that implement the Corridor Key
//  per-frame GPU pipeline: screen-colour rotation, downsample & normalisation
//  for neural inference, despill, alpha edge work (levels, gamma, erode/dilate,
//  blur), source passthrough, and final compositing. The final compose is a
//  render stage so the output goes straight to Final Cut Pro's destination
//  texture (which is a render target, not a shader-writable texture).
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

// MARK: - Normalisation for neural inference

/// Downsamples source and hint to the inference resolution and produces the
/// four-channel tensor the neural model expects (RGB mean/stddev normalised,
/// hint packed into alpha). The destination texture's dimensions set the
/// inference resolution; bilinear filtering handles the rescale.
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

    float3 normalized = (rgba.rgb - params.mean) * params.invStdDev;
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
            float gray = (r + newG + b) * (1.0 / 3.0);
            float fill = effectiveSpill * 0.5;
            r = r + fill * (gray / max(r, 1e-6));
            b = b + fill * (gray / max(b, 1e-6));
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

// MARK: - Morphology (separable erode / dilate)

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

// MARK: - Gaussian blur (separable)

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

// MARK: - Green screen detection & rough matte fallback

/// Coarse alpha-hint for the neural model. Matches the CorridorKey-Runtime
/// reference (`ColorUtils::generate_rough_matte` in
/// src/post_process/color_utils.cpp): the hint uses matte convention —
/// `1.0` means foreground/keep, `0.0` means screen/remove — so the trained
/// network reads it the same way it was trained. Sending raw greenness
/// here flips the model's output and the whole key inverts.
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

/// Produces a fallback alpha matte where 1 = foreground (opaque) and 0 =
/// green screen (transparent). Used when no MLX bridge is loaded. Matches
/// the CorridorKey-Runtime green-bias formula exactly so the fallback matte
/// and the MLX-ready hint line up pixel-for-pixel.
kernel void corridorKeyRoughMatteKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = source.read(gid);
    float greenBias = rgba.g - max(rgba.r, rgba.b);
    float alpha = 1.0 - saturate(greenBias * 2.0);
    destination.write(float4(alpha, 0.0, 0.0, 1.0), gid);
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

// MARK: - Resample (upscale / downscale)

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
