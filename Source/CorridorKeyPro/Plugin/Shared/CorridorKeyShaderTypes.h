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
    CKTextureIndexOutput = 6
} CorridorKeyTextureIndex;

// Fragment / compute argument buffer slots.
typedef enum CorridorKeyBufferIndex {
    CKBufferIndexDespillParams = 0,
    CKBufferIndexAlphaEdgeParams = 1,
    CKBufferIndexComposeParams = 2,
    CKBufferIndexScreenColorMatrix = 3,
    CKBufferIndexBlurWeights = 4,
    CKBufferIndexNormalizeParams = 5,
    CKBufferIndexSourcePassthroughParams = 6
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

typedef struct CKNormalizeParams {
    vector_float3 mean;
    vector_float3 invStdDev;
} CKNormalizeParams;

typedef struct CKSourcePassthroughParams {
    float erodeRadius; // In destination pixels.
    float blurRadius;
    float interiorThreshold;
} CKSourcePassthroughParams;

#endif /* CorridorKeyShaderTypes_h */
