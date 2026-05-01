//
//  CorridorKeyShaders.metal
//  CorridorKey by LateNite
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

// Sampler usage policy across the kernels in this file:
//
// * Kernels that resample (`corridorKeyNormalizeToBufferKernel`,
//   `corridorKeyCombineAndNormalizeKernel`, `corridorKeyExtractHintKernel`,
//   `corridorKeyComposeFragment`) keep `texture2d<…access::sample>` because
//   they need bilinear filtering on non-integer UVs.
// * Kernels that walk the texture in axis-aligned integer steps
//   (`corridorKey{ScreenMatrix,Despill,AlphaLevelsGamma,Morphology*,
//   Gaussian*,GreenHint,…}`) use `texture2d<…access::read>` — the
//   sampler hardware adds latency we don't need.
// Audit recorded after the kernel-fusion polish so future changes
// know which side of the line a given kernel sits on.

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
        case CKOutputModeHint:
            // Diagnostic: visualise the hint as bright red on a dim
            // dark-blue background. The dark-blue field proves the
            // compose path reached the screen even when the hint
            // happens to be all-zero — which is the symptom we'd
            // otherwise misread as "the diagnostic isn't running".
            // alpha = 1 → bright red; alpha = 0 → dark blue.
            return float4(alpha, 0.0, 0.15 * (1.0 - alpha), 1.0);
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

/// Corridor Key's despill runs in linear RGB. The screen colour is
/// passed in via `params.screenColor` — the kernel picks the dominant
/// channel (G for green, B for blue) and treats the other two as the
/// "limit" / "fill" pair, so the same code path despills both green
/// and blue screens without callers having to rotate the input first.
kernel void corridorKeyDespillKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKDespillParams &params [[buffer(CKBufferIndexDespillParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = source.read(gid);
    float3 rgb = rgba.rgb;

    // Identify the screen channel index from the canonical reference.
    // Green and Blue are the only supported screens in v1.0, but the
    // logic generalises to any single-channel-dominant reference (a
    // hypothetical red-screen pack would just need the right
    // `canonicalScreenReference` and the same code below would key it).
    const int screenIdx = (params.screenColor.b > params.screenColor.g) ? 2 : 1;
    const int otherA = (screenIdx == 1) ? 0 : 0; // R is always one of the non-screen channels for G/B screens.
    const int otherB = (screenIdx == 1) ? 2 : 1;

    // Ultra method — chroma-projection despill modelled on After
    // Effects' Advanced Spill Suppressor. Works in BT.709 YCbCr: the
    // screen colour sits at a fixed direction in CbCr space, so we
    // project each pixel onto that direction (computed from the
    // canonical screen colour, so the same kernel handles green and
    // blue) and subtract the positive component while keeping
    // luminance intact. Result: clean, flat removal that survives
    // soft edges and translucency without the over-bright mids the
    // Screen Subtract method introduces.
    if (params.method == CKSpillMethodUltra && params.strength > 0.0) {
        // BT.709 RGB → YCbCr (no offset; values are linear, not
        // studio-range, so we keep Cb/Cr signed around zero).
        const float3 luma = float3(0.2126, 0.7152, 0.0722);
        const float3 cbBasis = float3(-0.1146, -0.3854, 0.5000);
        const float3 crBasis = float3( 0.5000, -0.4542, -0.0458);
        float yLuma = dot(luma, rgb);
        float cb    = dot(cbBasis, rgb);
        float cr    = dot(crBasis, rgb);

        // Direction the screen colour points to in CbCr space.
        // Computed from the canonical reference each frame so the
        // kernel handles any colour pack without recompilation. The
        // length guard avoids a NaN when someone passes a neutral
        // grey reference (which has no chroma direction).
        float3 sc = params.screenColor;
        float2 chromaScreen = float2(dot(cbBasis, sc), dot(crBasis, sc));
        float screenChromaLen = length(chromaScreen);
        if (screenChromaLen > 1e-4) {
            float2 screenDir = chromaScreen / screenChromaLen;
            float2 chroma = float2(cb, cr);
            float screenProjection = dot(chroma, screenDir);

            // Only positive projections indicate spill — negative ones
            // mean the pixel is biased *away* from the screen colour
            // (red / orange / skin tones for green keys, warm tones
            // for blue keys) and we leave them alone.
            if (screenProjection > 0.0) {
                float removed = screenProjection * params.strength;
                float2 corrected = chroma - screenDir * removed;
                cb = corrected.x;
                cr = corrected.y;

                // YCbCr → BT.709 RGB (inverse of the matrix above).
                rgb.r = yLuma                + 1.5748 * cr;
                rgb.g = yLuma - 0.1873 * cb - 0.4681 * cr;
                rgb.b = yLuma + 1.8556 * cb;

                rgb = saturate(rgb);
            }
        }
        destination.write(float4(rgb, rgba.a), gid);
        return;
    }

    // Legacy methods (Average / DoubleLimit / Neutral / ScreenSubtract):
    // remove excess of the screen channel, distribute the removed
    // amount across the two non-screen channels. Generalises the
    // historical green-only kernel by indexing into `rgb` via the
    // resolved channel indices.
    float screenVal = rgb[screenIdx];
    float otherAVal = rgb[otherA];
    float otherBVal = rgb[otherB];

    float limit = 0.0;
    if (params.method == CKSpillMethodDoubleLimit) {
        limit = max(otherAVal, otherBVal);
    } else {
        limit = (otherAVal + otherBVal) * 0.5;
    }

    float spill = max(0.0, screenVal - limit);
    if (spill > 0.0 && params.strength > 0.0) {
        float effectiveSpill = spill * params.strength;
        float newScreenVal = screenVal - effectiveSpill;

        if (params.method == CKSpillMethodNeutral) {
            // Guard against division by near-zero on dark pixels. The
            // reference CorridorKey-Runtime uses 1e-3 (src/post_process/
            // despill.cpp) to avoid the noise amplification a 1e-6 floor
            // produced on dark skin tones / hair.
            float gray = (otherAVal + newScreenVal + otherBVal) * (1.0 / 3.0);
            float fill = effectiveSpill * 0.5;
            otherAVal = saturate(otherAVal + fill * (gray / max(otherAVal, 1e-3)));
            otherBVal = saturate(otherBVal + fill * (gray / max(otherBVal, 1e-3)));
        } else if (params.method == CKSpillMethodScreenSubtract) {
            // Keylight-style "screen subtract": treat the spill as a
            // contamination by the screen colour rather than just
            // excess of one channel. Subtract the spill weighted by
            // saturation so de-saturated regions (white shirts, hair
            // specular) don't get pushed magenta on green or yellow
            // on blue. This is the method most VFX artists default to
            // in Nuke / Fusion because it keeps the foreground's
            // neutral tones neutral.
            float maxRGB = max(max(otherAVal, newScreenVal), otherBVal);
            float minRGB = min(min(otherAVal, newScreenVal), otherBVal);
            float saturation = max(maxRGB - minRGB, 0.0);
            float saturatedSpill = effectiveSpill * saturation;
            otherAVal = saturate(otherAVal + saturatedSpill * 0.5);
            otherBVal = saturate(otherBVal + saturatedSpill * 0.5);
        } else {
            otherAVal = otherAVal + effectiveSpill * 0.5;
            otherBVal = otherBVal + effectiveSpill * 0.5;
        }
        rgb[screenIdx] = newScreenVal;
        rgb[otherA] = otherAVal;
        rgb[otherB] = otherBVal;
    }

    destination.write(float4(rgb, rgba.a), gid);
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

// MARK: - Morphology (separable erode / dilate — threadgroup-cached fast path)
//
// Each threadgroup loads a 1-D strip of (TG_WIDTH + 2*radius) pixels into
// threadgroup memory cooperatively, then every thread computes its output
// from the cached values. This replaces a `texture.sample()`-per-tap loop
// with `threadgroup` reads, which are ~10× lower latency than the sampler
// hardware path. Empirically beats the prior `sample`-based kernel by 3–5×
// at radius 5 on a 4K matte; MPS still wins for radii > kMorphMaxRadius
// because its running-window algorithm is O(1) per pixel regardless of r.
//
// `kMorphMaxRadius` caps the threadgroup-memory cost (TG_WIDTH + 2*r
// floats per group). 32 is enough for every user-facing morphology
// control after destination → inference scaling; bigger radii fall
// through to MPS in `MatteRefiner.applyMorphology`.

constant int kMorphTGWidth = 64;
constant int kMorphMaxRadius = 32;

kernel void corridorKeyMorphologyHorizontalKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant int &radius [[buffer(0)]],
    constant int &erode [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tg_thread [[thread_position_in_threadgroup]]
) {
    threadgroup float cache[kMorphTGWidth + 2 * kMorphMaxRadius];

    int width = int(destination.get_width());
    int height = int(destination.get_height());
    int tgOriginX = int(tg_id.x) * kMorphTGWidth;
    int y = int(gid.y);
    int rad = clamp(radius, 0, kMorphMaxRadius);
    int cacheCount = kMorphTGWidth + 2 * rad;

    int tx = int(tg_thread.x);
    if (y < height) {
        for (int i = tx; i < cacheCount; i += kMorphTGWidth) {
            int srcX = clamp(tgOriginX + i - rad, 0, width - 1);
            cache[i] = source.read(uint2(uint(srcX), uint(y))).r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (int(gid.x) >= width || y >= height) { return; }
    int localX = tx + rad;
    float best = cache[localX];
    for (int dx = 1; dx <= rad; ++dx) {
        float left = cache[localX - dx];
        float right = cache[localX + dx];
        if (erode != 0) {
            best = min(best, min(left, right));
        } else {
            best = max(best, max(left, right));
        }
    }
    destination.write(float4(best, 0.0, 0.0, 1.0), gid);
}

kernel void corridorKeyMorphologyVerticalKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant int &radius [[buffer(0)]],
    constant int &erode [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tg_thread [[thread_position_in_threadgroup]]
) {
    threadgroup float cache[kMorphTGWidth + 2 * kMorphMaxRadius];

    int width = int(destination.get_width());
    int height = int(destination.get_height());
    int tgOriginY = int(tg_id.y) * kMorphTGWidth;
    int x = int(gid.x);
    int rad = clamp(radius, 0, kMorphMaxRadius);
    int cacheCount = kMorphTGWidth + 2 * rad;

    int ty = int(tg_thread.y);
    if (x < width) {
        for (int i = ty; i < cacheCount; i += kMorphTGWidth) {
            int srcY = clamp(tgOriginY + i - rad, 0, height - 1);
            cache[i] = source.read(uint2(uint(x), uint(srcY))).r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (x >= width || int(gid.y) >= height) { return; }
    int localY = ty + rad;
    float best = cache[localY];
    for (int dy = 1; dy <= rad; ++dy) {
        float up = cache[localY - dy];
        float down = cache[localY + dy];
        if (erode != 0) {
            best = min(best, min(up, down));
        } else {
            best = max(best, max(up, down));
        }
    }
    destination.write(float4(best, 0.0, 0.0, 1.0), gid);
}

// MARK: - Gaussian blur (separable — threadgroup-cached fast path)
//
// Same threadgroup-strip pattern as the morphology kernels, but with a
// weighted sum instead of min/max. Note that `kernelRadius` here is the
// support radius from `MetalDeviceCacheEntry.gaussianWeightsBuffer`,
// which can exceed `kMorphMaxRadius`; for that case the kernel still
// works because the loop walks `weights[]` from the centre out, but the
// threadgroup cache is only sized for radii up to `kMorphMaxRadius` —
// so for larger sigmas we read straight from the texture for the
// out-of-cache taps. `MatteRefiner.applyGaussianBlur` already routes
// large sigmas (≥ 1.5) to MPS, so this kernel only ever sees small
// support that fits the cache.

kernel void corridorKeyGaussianHorizontalKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant float *weights [[buffer(CKBufferIndexBlurWeights)]],
    constant int &kernelRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tg_thread [[thread_position_in_threadgroup]]
) {
    threadgroup float cache[kMorphTGWidth + 2 * kMorphMaxRadius];

    int width = int(destination.get_width());
    int height = int(destination.get_height());
    int tgOriginX = int(tg_id.x) * kMorphTGWidth;
    int y = int(gid.y);
    int rad = clamp(kernelRadius, 0, kMorphMaxRadius);
    int cacheCount = kMorphTGWidth + 2 * rad;

    int tx = int(tg_thread.x);
    if (y < height) {
        for (int i = tx; i < cacheCount; i += kMorphTGWidth) {
            int srcX = clamp(tgOriginX + i - rad, 0, width - 1);
            cache[i] = source.read(uint2(uint(srcX), uint(y))).r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (int(gid.x) >= width || y >= height) { return; }
    int localX = tx + rad;
    float acc = cache[localX] * weights[0];
    for (int i = 1; i <= rad; ++i) {
        float w = weights[i];
        acc += (cache[localX - i] + cache[localX + i]) * w;
    }
    destination.write(float4(acc, 0.0, 0.0, 1.0), gid);
}

kernel void corridorKeyGaussianVerticalKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexMatte)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant float *weights [[buffer(CKBufferIndexBlurWeights)]],
    constant int &kernelRadius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tg_thread [[thread_position_in_threadgroup]]
) {
    threadgroup float cache[kMorphTGWidth + 2 * kMorphMaxRadius];

    int width = int(destination.get_width());
    int height = int(destination.get_height());
    int tgOriginY = int(tg_id.y) * kMorphTGWidth;
    int x = int(gid.x);
    int rad = clamp(kernelRadius, 0, kMorphMaxRadius);
    int cacheCount = kMorphTGWidth + 2 * rad;

    int ty = int(tg_thread.y);
    if (x < width) {
        for (int i = ty; i < cacheCount; i += kMorphTGWidth) {
            int srcY = clamp(tgOriginY + i - rad, 0, height - 1);
            cache[i] = source.read(uint2(uint(x), uint(srcY))).r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (x >= width || int(gid.y) >= height) { return; }
    int localY = ty + rad;
    float acc = cache[localY] * weights[0];
    for (int i = 1; i <= rad; ++i) {
        float w = weights[i];
        acc += (cache[localY - i] + cache[localY + i]) * w;
    }
    destination.write(float4(acc, 0.0, 0.0, 1.0), gid);
}

// MARK: - Shared hint-point structure
//
// Used by both the OSC overlay (`corridorKeyDrawOSCKernel`) and the
// pre-inference hint-rasterisation pass (`corridorKeyApplyHintPointsKernel`).
// Defined once here so both kernels see the same layout. `radius` is
// interpreted by each pass: object-normalised for hint rasterisation,
// OSC pixels for on-screen markers.

struct CKHintPoint {
    float x;
    float y;
    float radius;
    int kind; // -1 = hidden OSC slot, 0 = foreground, 1 = background
};

// MARK: - OSC overlay rendering
//
// Draws the user's hint points onto Final Cut Pro's OSC destination
// canvas. Foreground points and the subject marker are dark discs
// with a white person silhouette inside (matching the Standalone
// Editor's `Image(systemName: "person.fill")` marker); background
// points are rounded rectangles in the user's screen colour with a
// black border (matching the editor's screen-swatch marker). The OSC
// always renders over a transparent background — FCP composites the
// OSC layer on top of the source frame.
//
// Uses a render pass (vertex/fragment shaders), not a compute kernel,
// because FCP's OSC destination texture is render-target-only — it
// has `.renderTarget` usage but typically NOT `.shaderWrite`. This
// matches the FxShape sample's working pattern.

struct CKHintOSCRasterizerData {
    float4 clipSpacePosition [[position]];
    float2 textureCoordinate;
};

vertex CKHintOSCRasterizerData corridorKeyDrawOSCVertex(
    uint vertexID [[vertex_id]],
    constant CKVertex2D *vertices [[buffer(CKVertexInputIndexVertices)]],
    constant vector_uint2 *viewportSize [[buffer(CKVertexInputIndexViewportSize)]]
) {
    CKHintOSCRasterizerData out;
    float2 pixelPosition = vertices[vertexID].position;
    float2 viewport = float2(*viewportSize);
    out.clipSpacePosition.xy = pixelPosition / (viewport * 0.5);
    out.clipSpacePosition.z = 0.0;
    out.clipSpacePosition.w = 1.0;
    out.textureCoordinate = vertices[vertexID].textureCoordinate;
    return out;
}

/// Procedural "person.fill" silhouette — head + shoulders.
/// Returns a positive value outside the silhouette and a negative
/// value inside, suitable for a smoothstep-based AA edge. Tuned so
/// the proportions roughly match Apple's SF Symbol when the marker's
/// inner fill radius is 1.0 in the local space.
static inline float corridorKeyPersonSilhouetteSDF(float2 localPosition) {
    // Local coordinate convention: +Y is "down" (matching the OSC
    // canvas), origin at the marker centre, units = inner-fill radii.
    // Head: small circle slightly above centre.
    float2 headOffset = localPosition - float2(0.0, -0.34);
    float headRadius = 0.30;
    float headDistance = length(headOffset) - headRadius;

    // Shoulders / torso: a horizontally-wider rounded rectangle below
    // the head. Modelled as an SDF for a rounded box centred a touch
    // below the centre point.
    float2 bodyOffset = localPosition - float2(0.0, 0.30);
    float2 bodyHalfExtent = float2(0.55, 0.30);
    float bodyCornerRadius = 0.30;
    float2 bodyEdge = abs(bodyOffset) - bodyHalfExtent + bodyCornerRadius;
    float bodyDistance = length(max(bodyEdge, 0.0)) +
                         min(max(bodyEdge.x, bodyEdge.y), 0.0) -
                         bodyCornerRadius;

    // Smooth-min keeps the head and shoulders visually fused so the
    // silhouette reads as one figure rather than two stacked shapes.
    float k = 0.06;
    float h = clamp(0.5 + 0.5 * (bodyDistance - headDistance) / k, 0.0, 1.0);
    return mix(bodyDistance, headDistance, h) - k * h * (1.0 - h);
}

/// SDF for a rounded box used by the background marker. `localPosition`
/// is in inner-fill-radius units relative to the marker centre.
static inline float corridorKeyRoundedBoxSDF(
    float2 localPosition,
    float2 halfExtent,
    float cornerRadius
) {
    float2 q = abs(localPosition) - halfExtent + cornerRadius;
    return length(max(q, 0.0)) +
           min(max(q.x, q.y), 0.0) -
           cornerRadius;
}

fragment float4 corridorKeyDrawOSCFragment(
    CKHintOSCRasterizerData in [[stage_in]],
    constant CKHintPoint *points [[buffer(0)]],
    constant int &pointCount [[buffer(1)]],
    constant int &activePart [[buffer(2)]],
    constant float3 &screenColour [[buffer(3)]],
    constant vector_uint2 *viewportSize [[buffer(4)]]
) {
    // Texture coordinate is 0..1 across the OSC canvas, matching the
    // object-normalised space hint points are stored in. Marker shape
    // math is converted to OSC pixels before distance checks so a
    // circular badge stays circular in every FCP viewer aspect ratio.
    float2 uv = in.textureCoordinate;
    float2 viewport = max(float2(*viewportSize), float2(1.0));

    float4 colour = float4(0.0, 0.0, 0.0, 0.0);

    for (int i = 0; i < pointCount; ++i) {
        CKHintPoint p = points[i];
        if (p.kind < 0) { continue; }

        float2 offsetPixels = (uv - float2(p.x, p.y)) * viewport;
        // ~1 px wide anti-alias band in OSC pixels.
        float aaBand = max(max(fwidth(offsetPixels.x), fwidth(offsetPixels.y)) * 0.75, 0.75);

        float radius = max(p.radius, 1.0);
        bool isActive = ((i + 1) == activePart);

        if (p.kind == 1) {
            // Background hint — rounded rectangle in the user's
            // screen colour with a black outline. Mirrors the
            // standalone editor's `RoundedRectangle` background
            // marker (16x14 swatch, rounded 3px, 1.5px black border).
            // Local coordinates with Y-down so the canvas-space
            // marker reads "screen swatch sat at this point".
            float2 swatchHalfExtent = float2(radius * 0.90, radius * 0.78);
            float swatchCornerRadius = min(radius * 0.34, 3.0);
            float boxDistance = corridorKeyRoundedBoxSDF(
                offsetPixels,
                swatchHalfExtent,
                swatchCornerRadius
            );
            float fillAlpha = 1.0 - smoothstep(-aaBand, aaBand, boxDistance);
            float borderThickness = max(radius * 0.17, 1.0);
            float borderInner = boxDistance + borderThickness;
            float borderAlpha = (1.0 - smoothstep(-aaBand, aaBand, boxDistance))
                              * smoothstep(-aaBand, aaBand, borderInner);
            // Black border for the active marker brightens to the
            // FCP yellow used elsewhere so the user's drag target is
            // unambiguous against any background.
            float3 fillColour = clamp(screenColour, 0.0, 1.0);
            float3 borderColour = isActive
                ? float3(1.0, 0.85, 0.20)
                : float3(0.0);
            float3 swatchColour = mix(borderColour, fillColour, 1.0 - clamp(borderAlpha, 0.0, 1.0));
            float swatchAlpha = max(fillAlpha, borderAlpha);
            colour.rgb = mix(colour.rgb, swatchColour, swatchAlpha);
            colour.a = max(colour.a, swatchAlpha);
            continue;
        }

        // Foreground hint and subject marker. Dark filled disc with
        // white outline and a centred white person silhouette —
        // matches the standalone editor's `Image(systemName:
        // "person.fill")` over a circle marker so the visual language
        // is identical across the two surfaces.
        float distance = length(offsetPixels);
        float fillAlpha = 1.0 - smoothstep(radius - aaBand, radius + aaBand, distance);
        float strokeThickness = max(radius * 0.17, 1.0);
        float strokeInnerRadius = max(radius - strokeThickness, 0.0);
        float ringAlpha = fillAlpha
                        * smoothstep(strokeInnerRadius - aaBand, strokeInnerRadius + aaBand, distance);
        float3 fillColour = float3(0.10, 0.12, 0.18);
        float3 ringColour = isActive
            ? float3(1.0, 0.95, 0.40)
            : float3(1.0, 1.0, 1.0);

        // Render the silhouette inside the inner disc. The local
        // coordinate is the offset normalised by marker radius, so the
        // SDF is independent of the marker's size.
        float silhouetteScale = radius * 0.62;
        float2 localPosition = offsetPixels / max(silhouetteScale, 0.0001);
        float silhouetteSDF = corridorKeyPersonSilhouetteSDF(localPosition);
        // `aaBand` is in pixel units; convert to silhouette-local
        // coordinates so the AA band has consistent width regardless
        // of the marker radius.
        float silhouetteAA = max(aaBand / max(silhouetteScale, 0.0001), 0.04);
        float silhouetteAlpha = (1.0 - smoothstep(-silhouetteAA, silhouetteAA, silhouetteSDF))
                              * fillAlpha;

        // Layer order (back-to-front): outer ring → dark fill →
        // white silhouette inside the dark fill.
        float3 dotColour = mix(fillColour, ringColour, ringAlpha);
        // Silhouette punches white through the dark interior. The
        // active state keeps the silhouette white so it stays
        // legible against the brightened ring colour.
        dotColour = mix(dotColour, float3(1.0), silhouetteAlpha);

        float dotAlpha = max(fillAlpha, silhouetteAlpha);
        colour.rgb = mix(colour.rgb, dotColour, dotAlpha);
        colour.a = max(colour.a, dotAlpha);
    }
    return colour;
}

// MARK: - Hint-point rasterisation
//
// User-placed foreground / background hint points are stored in a
// CPU-side `HintPointSet` and uploaded to the GPU as a flat array of
// `(x, y, radius, kind)` tuples in object-normalised coords. This
// kernel expands each point into a soft-edged disc on top of the
// existing hint texture (which already carries either the Vision mask
// or the green-bias rough matte). Foreground points drive the hint
// toward 1.0; background points drive it toward 0.0.
//
// The fall-off uses `1 - smoothstep(...)` over the disc so the model
// sees a soft prior rather than a hard step — empirically the hint
// channel responds badly to discontinuities, but a smooth ramp lifts
// the matte without painting clipped artefacts.

kernel void corridorKeyApplyHintPointsKernel(
    texture2d<float, access::read_write> hint [[texture(CKTextureIndexOutput)]],
    constant CKHintPoint *points [[buffer(0)]],
    constant int &pointCount [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = hint.get_width();
    uint height = hint.get_height();
    if (gid.x >= width || gid.y >= height) { return; }

    // Object-normalised coordinates of the current pixel.
    float2 uv = (float2(gid) + 0.5) / float2(width, height);

    float current = hint.read(gid).r;
    float foregroundLift = 0.0;
    float backgroundPull = 0.0;

    for (int i = 0; i < pointCount; ++i) {
        CKHintPoint p = points[i];
        float2 offset = uv - float2(p.x, p.y);
        float distance = length(offset);
        float radius = max(p.radius, 0.001);
        // Smooth fall-off: 1 at the centre, 0 past the radius.
        float weight = 1.0 - smoothstep(0.0, radius, distance);
        if (p.kind == 0) {
            foregroundLift = max(foregroundLift, weight);
        } else {
            backgroundPull = max(backgroundPull, weight);
        }
    }

    // Foreground lift pulls the hint toward 1.0; background pull pulls
    // toward 0.0. Apply foreground first so a foreground point at the
    // same location wins ties — matches the user's mental model that
    // the most-recent click takes effect.
    float lifted = mix(current, 1.0, foregroundLift);
    float pulled = mix(lifted, 0.0, backgroundPull);
    hint.write(float4(pulled, 0.0, 0.0, 1.0), gid);
}

// MARK: - Chroma-prior alpha hint (rough-matte fallback)

/// Coarse alpha-hint for the neural model. Matches the CorridorKey-Runtime
/// reference (`ColorUtils::generate_rough_matte` in
/// src/post_process/color_utils.cpp): the hint uses matte convention —
/// `1.0` means foreground/keep, `0.0` means screen/remove — so the trained
/// network reads it the same way it was trained. Sending raw screen-
/// channel dominance flips the model's output and the whole key inverts.
///
/// `params.screenColor` selects which channel the kernel treats as the
/// screen — passing canonical green reproduces the v1.0 behaviour, passing
/// canonical blue keys blue-screen footage natively. Replaces the previous
/// `corridorKeyGreenHintKernel` / `roughMatte` kernels which hard-coded
/// the green channel and only worked because callers rotated blue input
/// into the green domain first.
kernel void corridorKeyChromaHintKernel(
    texture2d<float, access::read> source [[texture(CKTextureIndexSource)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    constant CKChromaHintParams &params [[buffer(CKBufferIndexChromaHintParams)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    float4 rgba = source.read(gid);
    float3 rgb = rgba.rgb;
    const int screenIdx = (params.screenColor.b > params.screenColor.g) ? 2 : 1;
    const int otherA = 0;
    const int otherB = (screenIdx == 1) ? 2 : 1;
    float screenBias = rgb[screenIdx] - max(rgb[otherA], rgb[otherB]);
    float matte = 1.0 - saturate(screenBias * 2.0);
    destination.write(float4(matte, 0.0, 0.0, 1.0), gid);
}

// MARK: - Hint union (max combine)

/// Combines two single-channel alpha-hint textures with `max(a, b)` so a
/// pixel that's "foreground" in either input stays foreground in the
/// output. Used when the Apple Vision hint is active and we want to fold
/// the chroma-prior in underneath — Vision picks up people / animals /
/// salient subjects, and the chroma prior fills in everything else
/// (props, set dressing, anything that isn't the screen colour).
/// Without the union, Vision-only mattes lose foreground props because
/// `VNGenerateForegroundInstanceMaskRequest` is a subject detector, not
/// a generic foreground segmenter.
kernel void corridorKeyHintUnionKernel(
    texture2d<float, access::read> hintA [[texture(CKTextureIndexHint)]],
    texture2d<float, access::read> hintB [[texture(CKTextureIndexCoarse)]],
    texture2d<float, access::write> destination [[texture(CKTextureIndexOutput)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 dims = uint2(destination.get_width(), destination.get_height());
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    float a = hintA.read(gid).r;
    float b = hintB.read(gid).r;
    destination.write(float4(max(a, b), 0.0, 0.0, 1.0), gid);
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

    // Stage 4: inverse screen-colour rotation. Identity for both
    // Green and Blue in the v1.0 native-bridge era — kept wired so a
    // future colour pack that *does* need a domain rotation can flip
    // the flag without re-introducing the `rgb` round-trip cost on
    // the common path.
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
