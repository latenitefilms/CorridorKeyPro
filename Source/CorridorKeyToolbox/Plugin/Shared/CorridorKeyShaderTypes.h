//
//  CorridorKeyShaderTypes.h
//  Corridor Key Toolbox
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
    CKTextureIndexLabel = 8
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
    CKBufferIndexCCLabelCounts = 11
} CorridorKeyBufferIndex;

// Mirrors the Swift `SpillMethod` enum.
typedef enum CorridorKeySpillMethod {
    CKSpillMethodAverage = 0,
    CKSpillMethodDoubleLimit = 1,
    CKSpillMethodNeutral = 2
} CorridorKeySpillMethod;

// Mirrors the Swift `OutputMode` enum.
typedef enum CorridorKeyOutputMode {
    CKOutputModeProcessed = 0,
    CKOutputModeMatteOnly = 1,
    CKOutputModeForegroundOnly = 2,
    CKOutputModeSourcePlusMatte = 3,
    CKOutputModeForegroundPlusMatte = 4
} CorridorKeyOutputMode;

// Vertex layout for the full-screen quad in pixel space.
typedef struct CKVertex2D {
    vector_float2 position;
    vector_float2 textureCoordinate;
} CKVertex2D;

// Per-frame parameter blocks. Kept tightly packed for efficient Metal uploads.
typedef struct CKDespillParams {
    float strength;
    int method; // CorridorKeySpillMethod
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
// is never touched. `screenColor` is the reference screen colour (green by
// default, rotated from blue via `ScreenColorEstimator`).
typedef struct CKEdgeDecontaminateParams {
    float strength;
    vector_float3 screenColor;
} CKEdgeDecontaminateParams;

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

#endif /* CorridorKeyShaderTypes_h */
