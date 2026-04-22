//
//  CorridorKeyShaders.metal
//  Corridor Key Pro
//
//  Metal kernels and vertex/fragment shaders that implement the Corridor Key
//  per-frame GPU pipeline: normalization for neural inference, despill, alpha
//  edge work (levels, gamma, erode/dilate, blur), source passthrough, screen
//  colour domain mapping, and final compositing.
//
//  All shaders operate in linear RGB unless noted otherwise. Texture reads use
//  explicit samplers so that results match on every Apple GPU architecture.
//

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

#include "CorridorKeyShaderTypes.h"

// MARK: - Full-screen quad rasteriser (shared vertex stage)

struct RasterizerData {
    float4 clipSpacePosition [[position]];
    float2 textureCoordinate;
};

vertex RasterizerData corridorKeyVertexShader(
    uint vertexID [[vertex_id]],
    constant CKVertex2D *vertices [[buffer(CKVertexInputIndexVertices)]],
    constant vector_uint2 *viewportSize [[buffer(CKVertexInputIndexViewportSize)]]
) {
    RasterizerData out;
    float2 pixelPosition = vertices[vertexID].position;
    float2 viewport = float2(*viewportSize);
    out.clipSpacePosition.xy = pixelPosition / (viewport * 0.5);
    out.clipSpacePosition.z = 0.0;
    out.clipSpacePosition.w = 1.0;
    out.textureCoordinate = vertices[vertexID].textureCoordinate;
    return out;
}

// Simple passthrough used when the host asks for a blit-style tile copy.
fragment float4 corridorKeyPassthroughFragment(
    RasterizerData in [[stage_in]],
    texture2d<float, access::sample> source [[texture(CKTextureIndexSource)]]
) {
    constexpr sampler bilinear(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    return source.sample(bilinear, in.textureCoordinate);
}

// MARK: - Screen colour domain mapping

// Applies a 3x3 colour matrix to the source texture. Used to normalise a blue
// screen into the green domain (and back again) so the neural model and despill
// path only ever see green.
fragment float4 corridorKeyApplyColorMatrixFragment(
    RasterizerData in [[stage_in]],
    texture2d<float, access::sample> source [[texture(CKTextureIndexSource)]],
    constant float3x3 &matrix [[buffer(CKBufferIndexScreenColorMatrix)]]
) {
    constexpr sampler bilinear(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float4 sample = source.sample(bilinear, in.textureCoordinate);
    float3 rgb = matrix * sample.rgb;
    return float4(rgb, sample.a);
}

// MARK: - Normalisation for neural inference

// Writes the four-channel tensor the neural model expects directly into a
// `.shared` destination texture. RGB is mean/stddev normalised (ImageNet
// statistics) and the hint is packed into the alpha plane. The destination
// texture's dimensions determine the inference resolution; both inputs are
// sampled with bilinear filtering so this kernel also performs the
// downscale from the source resolution.
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

// Corridor Key's despill runs in linear RGB. Green is assumed to be the screen
// colour; callers are expected to have rotated blue-screen content into the
// green domain before invoking this kernel.
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

// MARK: - Alpha levels, gamma, and compose

// Applies a levels remap and gamma curve to a single-channel alpha texture.
// The matte and foreground textures are kept in separate planes so that each
// pass remains trivially parallel.
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

// Horizontal pass of a separable min/max filter. Positive radii dilate,
// negative radii erode. The caller supplies the absolute radius as the buffer
// contents to avoid branching per sample.
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

// Vertical half of the same separable morphology filter.
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

// Separable horizontal Gaussian. Weights are pre-normalised on the host and
// include the centre tap at offset zero.
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

// Vertical Gaussian pass to complete the separable blur.
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

// MARK: - Rough matte fallback

// Generates a crude alpha-hint texture from the source by subtracting the
// strongest non-screen channel from the screen channel. Used when the user has
// not supplied an explicit hint clip, matching the CLI fallback behaviour.
kernel void corridorKeyRoughMatteKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = source.read(gid);
    float hint = max(0.0, rgba.g - max(rgba.r, rgba.b));
    destination.write(float4(hint, 0.0, 0.0, 1.0), gid);
}

// MARK: - Source passthrough blending

// Blends the original source RGB back into the despilled foreground, weighted
// by a feathered interior mask. Recovers texture that the neural model would
// otherwise smooth away inside fully opaque regions.
kernel void corridorKeySourcePassthroughKernel(
    texture2d<float, access::read> foreground [[texture(CKTextureIndexForeground)]],
    texture2d<float, access::read> sourceRGB [[texture(CKTextureIndexSource)]],
    texture2d<float, access::read> mask [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float m = clamp(mask.read(gid).r, 0.0, 1.0);
    float3 fg = foreground.read(gid).rgb;
    if (m <= 0.0) {
        destination.write(float4(fg, 1.0), gid);
        return;
    }
    float3 src = sourceRGB.read(gid).rgb;
    float3 blended = m * src + (1.0 - m) * fg;
    destination.write(float4(blended, 1.0), gid);
}

// MARK: - Final compose

// Combines the despilled foreground and refined matte into the output texture
// requested by the user: premultiplied composite, matte-only view, foreground
// only, source×matte, or foreground+matte.
kernel void corridorKeyComposeKernel(
    texture2d<float, access::read> foreground [[texture(CKTextureIndexForeground)]],
    texture2d<float, access::read> sourceRGB [[texture(CKTextureIndexSource)]],
    texture2d<float, access::read> matte [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKComposeParams &params [[buffer(CKBufferIndexComposeParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float alpha = clamp(matte.read(gid).r, 0.0, 1.0);
    float3 fg = foreground.read(gid).rgb;
    float3 src = sourceRGB.read(gid).rgb;
    float4 result;

    switch (params.outputMode) {
        case CKOutputModeMatteOnly:
            result = float4(alpha, alpha, alpha, 1.0);
            break;
        case CKOutputModeForegroundOnly:
            result = float4(fg, 1.0);
            break;
        case CKOutputModeSourcePlusMatte:
            result = float4(src * alpha, alpha);
            break;
        case CKOutputModeForegroundPlusMatte:
        case CKOutputModeProcessed:
        default:
            result = float4(fg * alpha, alpha);
            break;
    }

    destination.write(result, gid);
}

// MARK: - Downscale / upscale helpers

// Resamples a texture using hardware bilinear filtering. Works for both
// downscale (pre-inference) and upscale (post-inference) operations.
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

// Extracts the channel the user intends as a guide matte. Clips connected as
// RGBA contribute their alpha, Alpha-only clips contribute their sole channel,
// and RGB clips contribute their red value.
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
    // 0 = RGBA → use alpha, 1 = alpha only, 2 = RGB → use red.
    if (sourceLayout == 0) {
        hint = rgba.a;
    } else if (sourceLayout == 1) {
        hint = rgba.r;
    } else {
        hint = rgba.r;
    }
    destination.write(float4(hint, 0.0, 0.0, 1.0), gid);
}
