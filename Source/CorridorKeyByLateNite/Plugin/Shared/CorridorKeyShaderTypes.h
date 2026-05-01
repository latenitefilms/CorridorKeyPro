//
//  CorridorKeyShaderTypes.h
//  CorridorKey by LateNite
//
//  Shared type and index constants used by Metal shaders and the Swift host.
//

#ifndef CorridorKeyShaderTypes_h
#define CorridorKeyShaderTypes_h

#import <simd/simd.h>

// Vertex buffer indices for the full-screen textured quad pass.
typedef enum CorridorKeyVertexInputIndex {
    CKVertexInputIndexVertices = 0,
    CKVertexInputIndexViewportSize = 1
} CorridorKeyVertexInputIndex;

// Texture binding slots for the fragment and compute stages.
typedef enum CorridorKeyTextureIndex {
    CKTextureIndexSource = 0,
    CKTextureIndexMatte = 1,
    CKTextureIndexForeground = 2,
    CKTextureIndexHint = 3,
    CKTextureIndexTempA = 4,
    CKTextureIndexTempB = 5,
    CKTextureIndexOutput = 6,
    CKTextureIndexCoarse = 7,
    CKTextureIndexLabel = 8,
    // Temporal blend inputs: previous-frame alpha + previous-frame source.
    // Separated from the standard CKTextureIndexSource/Matte slots so a
    // single encoder can bind current and previous pairs at once.
    CKTextureIndexPreviousMatte = 9,
    CKTextureIndexPreviousSource = 10
} CorridorKeyTextureIndex;

// Fragment / compute argument buffer slots.
typedef enum CorridorKeyBufferIndex {
    CKBufferIndexDespillParams = 0,
    CKBufferIndexAlphaEdgeParams = 1,
    CKBufferIndexComposeParams = 2,
    CKBufferIndexScreenColorMatrix = 3,
    CKBufferIndexBlurWeights = 4,
    CKBufferIndexNormalizeParams = 5,
    CKBufferIndexSourcePassthroughParams = 6,
    CKBufferIndexRefinerParams = 7,
    CKBufferIndexLightWrapParams = 8,
    CKBufferIndexEdgeDecontaminateParams = 9,
    CKBufferIndexCCLabelParams = 10,
    CKBufferIndexCCLabelCounts = 11,
    CKBufferIndexTemporalBlendParams = 12,
    CKBufferIndexChromaHintParams = 13,
    CKBufferIndexSpillAlphaParams = 14
} CorridorKeyBufferIndex;

// Mirrors the Swift `SpillMethod` enum.
typedef enum CorridorKeySpillMethod {
    CKSpillMethodAverage = 0,
    CKSpillMethodDoubleLimit = 1,
    CKSpillMethodNeutral = 2,
    CKSpillMethodScreenSubtract = 3,
    CKSpillMethodUltra = 4
} CorridorKeySpillMethod;

// Mirrors the Swift `OutputMode` enum.
typedef enum CorridorKeyOutputMode {
    CKOutputModeProcessed = 0,
    CKOutputModeMatteOnly = 1,
    CKOutputModeForegroundOnly = 2,
    CKOutputModeSourcePlusMatte = 3,
    CKOutputModeForegroundPlusMatte = 4,
    /// Diagnostic: render the upstream alpha hint that MLX reads as its
    /// 4th input channel (Vision mask, OSC overlays, or green-bias).
    /// Visualised as red on black so the user can spot misalignment
    /// between the hint and the subject before MLX touches it.
    CKOutputModeHint = 5
} CorridorKeyOutputMode;

// Vertex layout for the full-screen quad in pixel space.
typedef struct CKVertex2D {
    vector_float2 position;
    vector_float2 textureCoordinate;
} CKVertex2D;

// Per-frame parameter blocks. Kept tightly packed for efficient Metal uploads.
//
// `screenColor` is the canonical reference for the keyed screen colour
// in linear RGB (e.g. `(0.08, 0.84, 0.08)` for green or `(0.08, 0.16,
// 0.84)` for blue). The kernel reads it to identify which channel is
// the screen channel — every spill method generalises to "remove
// excess of the screen channel and rebalance across the two non-screen
// channels", so a single kernel handles both colours without branching
// on a separate `screenColor` enum.
typedef struct CKDespillParams {
    float strength;
    int method; // CorridorKeySpillMethod
    vector_float3 screenColor;
} CKDespillParams;

typedef struct CKAlphaEdgeParams {
    float blackPoint;
    float whitePoint;
    float gamma;
    float morphRadius;   // Positive dilates, negative erodes. In destination pixels.
    float blurRadius;    // In destination pixels; zero skips the pass.
} CKAlphaEdgeParams;

typedef struct CKComposeParams {
    int outputMode; // CorridorKeyOutputMode
} CKComposeParams;

// Normalisation parameters for the neural input tensor. The working-space
// matrix maps whatever colour space the host handed us (Rec.709, Rec.2020,
// Display P3 linear, etc.) into the Rec.709-linear-sRGB space the model was
// trained on, so the model sees consistent values regardless of project gamut.
typedef struct CKNormalizeParams {
    simd_float3x3 workingToRec709;
    vector_float3 mean;
    vector_float3 invStdDev;
} CKNormalizeParams;

typedef struct CKSourcePassthroughParams {
    float erodeRadius; // In destination pixels.
    float blurRadius;
    float interiorThreshold;
} CKSourcePassthroughParams;

// Refiner-strength blend parameters. `strength` = 1.0 passes the model's
// refined alpha through unchanged; `< 1.0` biases toward the blurred
// "coarse" stand-in (softer edges); `> 1.0` extrapolates toward sharper
// edges (clamped to [0, 1] afterwards).
typedef struct CKRefinerParams {
    float strength;
} CKRefinerParams;

// Light-wrap parameters. `strength` mixes wrap colour into the foreground
// along `(1 - matte)` falloff. `edgeBias` biases toward the matte boundary
// — zero = full wrap across transparent zones, higher values = only a thin
// ring near the edges.
typedef struct CKLightWrapParams {
    float strength;
    float edgeBias;
} CKLightWrapParams;

// Edge colour decontamination parameters. Subtracts screen-colour residual
// from the foreground RGB, weighted by `(1 - matte)` so the opaque interior
// is never touched. `screenColor` is the canonical reference for the keyed
// screen — green or blue, picked from `ScreenColor.canonicalScreenReference`
// in Swift.
typedef struct CKEdgeDecontaminateParams {
    float strength;
    vector_float3 screenColor;
} CKEdgeDecontaminateParams;

// Chroma-prior alpha-hint generator parameters. The hint is a single-channel
// matte derived from the source RGB, biased to produce `0` where the screen
// colour dominates and `1` where the foreground does. `screenColor` selects
// which channel the kernel treats as the screen — passing canonical green
// reproduces the v1.0 behaviour, passing canonical blue keys blue-screen
// footage without rotation.
typedef struct CKChromaHintParams {
    vector_float3 screenColor;
} CKChromaHintParams;

// Spill-driven alpha attenuation parameters. The matte refinement stage
// produces clean alpha for regions the network is confident about, but
// model errors leave faint matte values in the screen background — and
// the colour despill turns those errors into a coloured halo (blue
// artefacts at strength 0, yellow / greenish artefacts at strength 1)
// because the despilled foreground colour ends up multiplied by a
// non-zero alpha at compose time. This stage reads the foreground's
// chroma projection onto the screen direction and pulls alpha toward
// zero proportional to that projection × `strength`, so the user's
// "Despill Strength" slider drives both colour cleanup and matte
// transparency together. Mirrors the original CorridorKey reference's
// premultiplied-output convention (`processed = despilled * alpha`)
// while still allowing the user to opt out by setting strength to 0.
typedef struct CKSpillAlphaParams {
    float strength;
    vector_float3 screenColor;
} CKSpillAlphaParams;

// Connected-components despeckle parameters.
// * `areaThreshold` — a component is preserved if its pixel count is at or
//   above this value, zeroed otherwise.
// * `matteThreshold` — threshold used to binarise the matte into the label
//   texture at the init stage (0.5 by default).
// * `labelSpan` — number of tiles along each axis in the label texture; the
//   kernel multiplies coordinates by this to derive a unique integer label
//   per pixel.
typedef struct CKCCLabelParams {
    int areaThreshold;
    int labelSpan;
    float matteThreshold;
    float blurSigma;
} CKCCLabelParams;

// Fused matte-refine params: levels + gamma + refiner blend in one pass.
// `refinerStrength == 1` skips the refiner blend (no coarse read). `gamma
// == 1` skips the gamma pow() call. Each combine reduces per-kernel
// overhead relative to running three separate dispatches.
typedef struct CKMatteRefineParams {
    float blackPoint;
    float whitePoint;
    float gamma;
    float refinerStrength;
} CKMatteRefineParams;

// Fused foreground post-process params: source passthrough + light wrap +
// edge decontamination + inverse screen matrix in one pass. The enable
// flags let callers avoid the downstream reads/work when a stage is off
// — the GPU branches coherently across all threads so these cost almost
// nothing when disabled.
typedef struct CKForegroundPostProcessParams {
    simd_float3x3 inverseScreenMatrix;
    vector_float3 screenColor;
    float lightWrapStrength;
    float lightWrapEdgeBias;
    float edgeDecontaminateStrength;
    int sourcePassthroughEnabled;
    int lightWrapEnabled;
    int edgeDecontaminateEnabled;
    int applyInverseRotation;
} CKForegroundPostProcessParams;

// Temporal-blend parameters. The kernel runs *during analysis* over a
// sequence of inference-resolution alpha textures, reading the previous
// frame's alpha and source and blending the current frame's alpha toward
// the previous value on pixels where the RGB has barely changed.
//
// `strength` is the blend weight when the pixel is deemed stationary
// (`motion == 0`). `motionThreshold` is the max-channel absolute RGB
// delta at which the gate reaches zero blend — values above imply real
// motion, so the kernel passes the current alpha through unchanged.
// Linear falloff between 0 and 2×threshold.
typedef struct CKTemporalBlendParams {
    float strength;
    float motionThreshold;
} CKTemporalBlendParams;

#endif /* CorridorKeyShaderTypes_h */
