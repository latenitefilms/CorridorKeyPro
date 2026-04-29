//
//  ParameterEnumerations.swift
//  CorridorKey by LateNite
//
//  Swift-native representations of each choice parameter. The `rawValue` lines
//  up with the `addPopupMenu` index order used in
//  `CorridorKeyToolboxPlugIn+Parameters.swift`; changing that order here will
//  invalidate saved user documents.
//

import Foundation

/// Colour of the screen being keyed out. Blue is rotated into the green domain
/// before inference and despill, then rotated back for output.
public enum ScreenColor: Int, Sendable, CaseIterable, Codable {
    case green = 0
    case blue = 1

    public var displayName: String {
        switch self {
        case .green: return "Green"
        case .blue: return "Blue"
        }
    }
}

/// How the alpha hint is generated for the MLX bridge's 4th input
/// channel. Replaces the earlier "Auto Subject Hint" toggle so users
/// can explicitly opt into Apple Vision (best-quality but binary) or
/// the green-bias prior (fastest, soft gradient) without reading the
/// release notes to know what the toggle does. The new `manual`
/// option short-circuits both upstream priors and feeds only the
/// user-placed hint dots — useful for shots Vision struggles with
/// (e.g. heavy motion blur, partially occluded subjects).
public enum HintMode: Int, Sendable, CaseIterable, Codable {
    /// Fast green / blue chroma prior. The MLX network is trained on
    /// this style of soft, gradient hint and tends to produce its
    /// best results here when the screen colour itself is clean.
    case automatic = 0
    /// Apple Vision foreground-instance mask, run on the Neural
    /// Engine. Beats the chroma prior on most footage because it
    /// segments by subject saliency rather than by green-channel
    /// dominance.
    case appleVision = 1
    /// Same upstream chroma prior as `.automatic`, but the user's
    /// hint dots are required — the renderer fails the analysis if
    /// the hint set is empty. Initial design ran the inference on
    /// a zero base + dots only, but the MLX network was trained on
    /// a soft chroma-derived gradient or a Vision-style binary
    /// mask; sparse dots on black sit outside that distribution
    /// and produce nonsense mattes. Falling back to the chroma
    /// prior with the user's dots overlaid gives the network input
    /// it understands while keeping the "you have to place hints"
    /// semantics this mode is named for.
    case manual = 2

    public var displayName: String {
        switch self {
        case .automatic: return "Automatic"
        case .appleVision: return "Apple Vision Framework"
        case .manual: return "Manual Hint"
        }
    }

    /// Whether this mode runs the Vision foreground request when
    /// computing the upstream hint.
    public var usesVisionPrior: Bool {
        self == .appleVision
    }

    /// Whether this mode requires at least one user-placed hint
    /// point in the render request. The render path checks this at
    /// the start of the pre-inference stage and returns a
    /// pass-through frame plus a status message rather than feeding
    /// MLX an empty hint.
    public var requiresUserHints: Bool {
        self == .manual
    }
}

/// Quality rung used for neural inference. `automatic` chooses a safe tier
/// based on the input resolution and the available physical memory.
public enum QualityMode: Int, Sendable, CaseIterable, Codable {
    case automatic = 0
    case draft512 = 1
    case high1024 = 2
    case ultra1536 = 3
    case maximum2048 = 4

    /// RAM-aware rung ceiling for the `.automatic` mode. The Hiera-based
    /// MLX bridge needs ~16 GB of working set at 1024 and ~64 GB at 2048
    /// (one frame plus MLX's per-call buffer cache). On a 32 GB Mac the
    /// 2048 path forces ~2 minutes of swap per frame; on a 64 GB Mac
    /// 2048 is borderline. Capping `automatic` keeps users at a rung the
    /// machine can serve in real time. Users who want to push past the
    /// cap can pick the explicit `.maximum2048` etc. modes — those skip
    /// this ceiling.
    public static func automaticInferenceCeiling(physicalMemoryBytes: UInt64) -> Int {
        let oneGigabyte: UInt64 = 1 << 30
        if physicalMemoryBytes >= 96 * oneGigabyte { return 2048 }
        if physicalMemoryBytes >= 64 * oneGigabyte { return 1536 }
        return 1024
    }

    /// Full ladder of available rungs, in ascending order. Used by the
    /// device capability cache to walk upward from the static ceiling
    /// when looking for an empirically-faster rung.
    public static let inferenceLadder: [Int] = [512, 768, 1024, 1536, 2048]

    /// Returns the inference resolution that should be used for a given input
    /// frame long-edge. Mirrors the automatic rung mapping used by the OFX
    /// plugin so that behaviour matches across hosts. The `.automatic` path
    /// also clamps to the host's RAM-aware ceiling so a 4K clip on a
    /// 32 GB Mac doesn't accidentally pick the 2048 rung and spend the
    /// next several minutes in swap.
    public func resolvedInferenceResolution(forLongEdge longEdgePixels: Int) -> Int {
        resolvedInferenceResolution(
            forLongEdge: longEdgePixels,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
        )
    }

    /// Returns the inference resolution including any device-capability
    /// upgrade observed by `DeviceCapabilityCache`. On a fast Mac that
    /// has previously run inference at a rung above the static RAM
    /// ceiling under the real-time budget, the `automatic` path lifts
    /// to that rung instead of the conservative default.
    public func resolvedInferenceResolution(
        forLongEdge longEdgePixels: Int,
        deviceRegistryID: UInt64
    ) -> Int {
        resolvedInferenceResolution(
            forLongEdge: longEdgePixels,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            deviceRegistryID: deviceRegistryID,
            cache: DeviceCapabilityCache.shared
        )
    }

    /// Test seam. Production callers use `resolvedInferenceResolution(forLongEdge:)`
    /// which reads `ProcessInfo.processInfo.physicalMemory`; tests pass an
    /// explicit value so the mapping is deterministic across machines.
    public func resolvedInferenceResolution(
        forLongEdge longEdgePixels: Int,
        physicalMemoryBytes: UInt64
    ) -> Int {
        resolvedInferenceResolution(
            forLongEdge: longEdgePixels,
            physicalMemoryBytes: physicalMemoryBytes,
            deviceRegistryID: nil,
            cache: nil
        )
    }

    /// Full-resolution implementation. The cache lift only applies to
    /// `.automatic`; explicit rungs always honour the user's choice
    /// (those skip the RAM ceiling too).
    public func resolvedInferenceResolution(
        forLongEdge longEdgePixels: Int,
        physicalMemoryBytes: UInt64,
        deviceRegistryID: UInt64?,
        cache: DeviceCapabilityCache?
    ) -> Int {
        switch self {
        case .automatic:
            let preferred: Int
            switch longEdgePixels {
            case ...1000: preferred = 512
            case 1001...2000: preferred = 1024
            case 2001...3000: preferred = 1536
            default: preferred = 2048
            }
            let staticCeiling = Self.automaticInferenceCeiling(physicalMemoryBytes: physicalMemoryBytes)
            let liftedCeiling: Int
            if let cache, let deviceRegistryID {
                liftedCeiling = cache.recommendedCeiling(
                    deviceRegistryID: deviceRegistryID,
                    staticCeiling: staticCeiling,
                    ladder: Self.inferenceLadder
                )
            } else {
                liftedCeiling = staticCeiling
            }
            return min(preferred, liftedCeiling)
        case .draft512: return 512
        case .high1024: return 1024
        case .ultra1536: return 1536
        case .maximum2048: return 2048
        }
    }

    public var displayName: String {
        switch self {
        case .automatic: return "Recommended"
        case .draft512: return "Draft (512)"
        case .high1024: return "High (1024)"
        case .ultra1536: return "Ultra (1536)"
        case .maximum2048: return "Maximum (2048)"
        }
    }
}

/// Despill redistribution strategy. Mirrors `SpillMethod` in the C++ reference
/// and the `CorridorKeySpillMethod` enum on the shader side.
public enum SpillMethod: Int, Sendable, CaseIterable, Codable {
    case average = 0
    case doubleLimit = 1
    case neutral = 2
    case screenSubtract = 3
    /// Advanced Ultra-style despill — projects each pixel's chroma
    /// onto the screen-colour direction in YCbCr space and subtracts
    /// that projection while preserving luminance. Removes the soft
    /// chroma fringe that the Average / Double Limit paths leave on
    /// hair and feathered edges, without the over-bright mids the
    /// Screen Subtract path can introduce on translucent fabric.
    /// Modelled on After Effects' Advanced Spill Suppressor.
    case ultra = 4

    public var shaderValue: Int32 { Int32(rawValue) }

    public var displayName: String {
        switch self {
        case .average: return "Average"
        case .doubleLimit: return "Double Limit"
        case .neutral: return "Neutral"
        case .screenSubtract: return "Screen Subtract"
        case .ultra: return "Ultra (Chroma Project)"
        }
    }
}

/// How the final composite is assembled and written to Final Cut Pro.
public enum OutputMode: Int, Sendable, CaseIterable, Codable {
    case processed = 0
    case matteOnly = 1
    case foregroundOnly = 2
    case sourcePlusMatte = 3
    case foregroundPlusMatte = 4
    /// Diagnostic mode: render the alpha *hint* the MLX bridge sees as
    /// its 4th input channel. Lets the user verify the upstream hint
    /// (Vision subject mask, OSC dots, or the green-bias rough matte)
    /// before MLX inference is involved. Visualised as red-on-black:
    /// red = "this is foreground" hint, black = "this is screen".
    case hint = 5

    public var shaderValue: Int32 { Int32(rawValue) }

    public var displayName: String {
        switch self {
        case .processed: return "Processed"
        case .matteOnly: return "Matte Only"
        case .foregroundOnly: return "Foreground Only"
        case .sourcePlusMatte: return "Source + Matte"
        case .foregroundPlusMatte: return "Foreground + Matte"
        case .hint: return "Hint (Diagnostic)"
        }
    }
}

/// Kernel used to resample between destination and inference resolutions.
public enum UpscaleMethod: Int, Sendable, CaseIterable, Codable {
    case bilinear = 0
    case lanczos = 1

    public var displayName: String {
        switch self {
        case .bilinear: return "Bilinear"
        case .lanczos: return "Lanczos"
        }
    }
}
