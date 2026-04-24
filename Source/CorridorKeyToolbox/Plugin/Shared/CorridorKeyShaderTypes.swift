//
//  CorridorKeyShaderTypes.swift
//  Corridor Key Toolbox
//
//  Swift mirror of `CorridorKeyShaderTypes.h`. The Xcode FxPlug target
//  imports the C header via its bridging header; the Swift Package build —
//  which drives headless Metal stage tests — has no bridging header, so
//  this file is what Swift sees when the SPM `CorridorKeyToolboxMetalStages`
//  target compiles. Every declaration here must byte-identically match the
//  C header; if you add or rename a type in one place, update the other
//  in the same commit.
//

#if CORRIDOR_KEY_SPM_MIRROR

import Foundation
import simd

// MARK: - Index enums

enum CorridorKeyVertexInputIndex: Int32 {
    case CKVertexInputIndexVertices = 0
    case CKVertexInputIndexViewportSize = 1
}

let CKVertexInputIndexVertices = CorridorKeyVertexInputIndex.CKVertexInputIndexVertices
let CKVertexInputIndexViewportSize = CorridorKeyVertexInputIndex.CKVertexInputIndexViewportSize

enum CorridorKeyTextureIndex: Int32 {
    case CKTextureIndexSource = 0
    case CKTextureIndexMatte = 1
    case CKTextureIndexForeground = 2
    case CKTextureIndexHint = 3
    case CKTextureIndexTempA = 4
    case CKTextureIndexTempB = 5
    case CKTextureIndexOutput = 6
    case CKTextureIndexCoarse = 7
    case CKTextureIndexLabel = 8
    case CKTextureIndexPreviousMatte = 9
    case CKTextureIndexPreviousSource = 10
}

let CKTextureIndexSource = CorridorKeyTextureIndex.CKTextureIndexSource
let CKTextureIndexMatte = CorridorKeyTextureIndex.CKTextureIndexMatte
let CKTextureIndexForeground = CorridorKeyTextureIndex.CKTextureIndexForeground
let CKTextureIndexHint = CorridorKeyTextureIndex.CKTextureIndexHint
let CKTextureIndexTempA = CorridorKeyTextureIndex.CKTextureIndexTempA
let CKTextureIndexTempB = CorridorKeyTextureIndex.CKTextureIndexTempB
let CKTextureIndexOutput = CorridorKeyTextureIndex.CKTextureIndexOutput
let CKTextureIndexCoarse = CorridorKeyTextureIndex.CKTextureIndexCoarse
let CKTextureIndexLabel = CorridorKeyTextureIndex.CKTextureIndexLabel
let CKTextureIndexPreviousMatte = CorridorKeyTextureIndex.CKTextureIndexPreviousMatte
let CKTextureIndexPreviousSource = CorridorKeyTextureIndex.CKTextureIndexPreviousSource

enum CorridorKeyBufferIndex: Int32 {
    case CKBufferIndexDespillParams = 0
    case CKBufferIndexAlphaEdgeParams = 1
    case CKBufferIndexComposeParams = 2
    case CKBufferIndexScreenColorMatrix = 3
    case CKBufferIndexBlurWeights = 4
    case CKBufferIndexNormalizeParams = 5
    case CKBufferIndexSourcePassthroughParams = 6
    case CKBufferIndexRefinerParams = 7
    case CKBufferIndexLightWrapParams = 8
    case CKBufferIndexEdgeDecontaminateParams = 9
    case CKBufferIndexCCLabelParams = 10
    case CKBufferIndexCCLabelCounts = 11
    case CKBufferIndexTemporalBlendParams = 12
}

let CKBufferIndexDespillParams = CorridorKeyBufferIndex.CKBufferIndexDespillParams
let CKBufferIndexAlphaEdgeParams = CorridorKeyBufferIndex.CKBufferIndexAlphaEdgeParams
let CKBufferIndexComposeParams = CorridorKeyBufferIndex.CKBufferIndexComposeParams
let CKBufferIndexScreenColorMatrix = CorridorKeyBufferIndex.CKBufferIndexScreenColorMatrix
let CKBufferIndexBlurWeights = CorridorKeyBufferIndex.CKBufferIndexBlurWeights
let CKBufferIndexNormalizeParams = CorridorKeyBufferIndex.CKBufferIndexNormalizeParams
let CKBufferIndexSourcePassthroughParams = CorridorKeyBufferIndex.CKBufferIndexSourcePassthroughParams
let CKBufferIndexRefinerParams = CorridorKeyBufferIndex.CKBufferIndexRefinerParams
let CKBufferIndexLightWrapParams = CorridorKeyBufferIndex.CKBufferIndexLightWrapParams
let CKBufferIndexEdgeDecontaminateParams = CorridorKeyBufferIndex.CKBufferIndexEdgeDecontaminateParams
let CKBufferIndexCCLabelParams = CorridorKeyBufferIndex.CKBufferIndexCCLabelParams
let CKBufferIndexCCLabelCounts = CorridorKeyBufferIndex.CKBufferIndexCCLabelCounts
let CKBufferIndexTemporalBlendParams = CorridorKeyBufferIndex.CKBufferIndexTemporalBlendParams

// Spill method and output mode enums. Use Int32 so rawValue matches the
// shader's signed int.

enum CorridorKeySpillMethod: Int32 {
    case CKSpillMethodAverage = 0
    case CKSpillMethodDoubleLimit = 1
    case CKSpillMethodNeutral = 2
}

let CKSpillMethodAverage = CorridorKeySpillMethod.CKSpillMethodAverage
let CKSpillMethodDoubleLimit = CorridorKeySpillMethod.CKSpillMethodDoubleLimit
let CKSpillMethodNeutral = CorridorKeySpillMethod.CKSpillMethodNeutral

enum CorridorKeyOutputMode: Int32 {
    case CKOutputModeProcessed = 0
    case CKOutputModeMatteOnly = 1
    case CKOutputModeForegroundOnly = 2
    case CKOutputModeSourcePlusMatte = 3
    case CKOutputModeForegroundPlusMatte = 4
}

let CKOutputModeProcessed = CorridorKeyOutputMode.CKOutputModeProcessed
let CKOutputModeMatteOnly = CorridorKeyOutputMode.CKOutputModeMatteOnly
let CKOutputModeForegroundOnly = CorridorKeyOutputMode.CKOutputModeForegroundOnly
let CKOutputModeSourcePlusMatte = CorridorKeyOutputMode.CKOutputModeSourcePlusMatte
let CKOutputModeForegroundPlusMatte = CorridorKeyOutputMode.CKOutputModeForegroundPlusMatte

// MARK: - Param structs

struct CKVertex2D {
    var position: SIMD2<Float>
    var textureCoordinate: SIMD2<Float>

    init(position: SIMD2<Float>, textureCoordinate: SIMD2<Float>) {
        self.position = position
        self.textureCoordinate = textureCoordinate
    }
}

struct CKDespillParams {
    var strength: Float
    var method: Int32
}

struct CKAlphaEdgeParams {
    var blackPoint: Float
    var whitePoint: Float
    var gamma: Float
    var morphRadius: Float
    var blurRadius: Float
}

struct CKComposeParams {
    var outputMode: Int32
}

struct CKNormalizeParams {
    var workingToRec709: simd_float3x3
    var mean: SIMD3<Float>
    var invStdDev: SIMD3<Float>
}

struct CKSourcePassthroughParams {
    var erodeRadius: Float
    var blurRadius: Float
    var interiorThreshold: Float
}

struct CKRefinerParams {
    var strength: Float
}

struct CKLightWrapParams {
    var strength: Float
    var edgeBias: Float
}

struct CKEdgeDecontaminateParams {
    var strength: Float
    var screenColor: SIMD3<Float>
}

struct CKCCLabelParams {
    var areaThreshold: Int32
    var labelSpan: Int32
    var matteThreshold: Float
    var blurSigma: Float
}

struct CKMatteRefineParams {
    var blackPoint: Float
    var whitePoint: Float
    var gamma: Float
    var refinerStrength: Float
}

struct CKForegroundPostProcessParams {
    var inverseScreenMatrix: simd_float3x3
    var screenColor: SIMD3<Float>
    var lightWrapStrength: Float
    var lightWrapEdgeBias: Float
    var edgeDecontaminateStrength: Float
    var sourcePassthroughEnabled: Int32
    var lightWrapEnabled: Int32
    var edgeDecontaminateEnabled: Int32
    var applyInverseRotation: Int32
}

struct CKTemporalBlendParams {
    var strength: Float
    var motionThreshold: Float
}

#endif
