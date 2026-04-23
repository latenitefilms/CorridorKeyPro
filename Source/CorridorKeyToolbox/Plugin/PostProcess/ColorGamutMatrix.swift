//
//  ColorGamutMatrix.swift
//  Corridor Key Toolbox
//
//  Builds 3x3 linear matrices that map Final Cut Pro's working colour gamut
//  into the Rec.709-linear sRGB space the neural network was trained on, so
//  the model sees consistent values whatever gamut the user is editing in.
//
//  FxPlug exposes only two gamut options through `FxColorGamutAPI_v2`:
//  Rec.709 and Rec.2020 (see `FxColorPrimaries` in FxTypes.h). SDR projects
//  and Display P3 HDR both report Rec.709 primaries at this level of the
//  API — the host's colour management layer handles the primaries-to-
//  Rec.709 mapping for us in that case. HDR Rec.2020 projects need the
//  explicit transform below.
//
//  Values come straight from ITU-R BT.2087 "Colour conversion from
//  Recommendation ITU-R BT.709 to Recommendation ITU-R BT.2020 colorimetry".
//

import Foundation
import simd

/// Wrapper around a 3x3 linear transform that takes the host's working-space
/// RGB to Rec.709 linear RGB.
struct WorkingSpaceTransform: Sendable, Equatable {
    /// Column-major 3x3 matrix. Multiplying on the right of `workingToRec709`
    /// transforms a working-space RGB vector into Rec.709 linear RGB.
    let workingToRec709: simd_float3x3

    /// Identity — used for projects already working in Rec.709 / sRGB linear
    /// (which covers nearly all SDR FCP timelines).
    static let identity = WorkingSpaceTransform(
        workingToRec709: matrix_identity_float3x3
    )
}

/// The FxPlug colour gamuts Corridor Key Toolbox will encounter. We map the
/// `FxColorPrimaries` enum to this Swift type at the render boundary so the
/// rest of the pipeline can reason in Swift-native terms.
enum WorkingColorGamut: Int, Sendable, CaseIterable, Codable {
    /// Rec.709 / sRGB primaries (SDR default plus Display P3 as reported
    /// through this particular FCP API, which only distinguishes Rec.709
    /// and Rec.2020 at the primary level).
    case rec709
    /// Rec.2020 primaries (HDR timelines).
    case rec2020
}

enum ColorGamutMatrix {

    /// Returns the working-space transform for a given gamut.
    static func transform(for gamut: WorkingColorGamut) -> WorkingSpaceTransform {
        switch gamut {
        case .rec709:
            return .identity

        case .rec2020:
            // Rec.2020 → Rec.709 linear. From ITU-R BT.2087.
            return WorkingSpaceTransform(
                workingToRec709: simd_float3x3(
                    SIMD3<Float>( 1.6605, -0.1246, -0.0182),
                    SIMD3<Float>(-0.5876,  1.1329, -0.1006),
                    SIMD3<Float>(-0.0728, -0.0083,  1.1187)
                )
            )
        }
    }

    /// Maps a raw `FxColorPrimaries` (imported as `NSUInteger` / `UInt`) to
    /// our Swift gamut enum. Callers pass the value returned from
    /// `FxColorGamutAPI_v2.colorPrimaries()` through this bridge so the rest
    /// of the pipeline never touches FxPlug types directly.
    static func gamut(fromColorPrimariesRaw raw: UInt) -> WorkingColorGamut {
        // `kFxColorPrimaries_Rec2020` is the second (and currently final)
        // value of the enum, so any non-zero raw is Rec.2020 today.
        raw == 0 ? .rec709 : .rec2020
    }
}
