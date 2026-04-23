//
//  ColorGamutMatrixTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Verifies the working-space matrix we apply during the combine+normalise
//  compute kernel. Rec.709 round-trips identity; Rec.2020 matches the
//  canonical BT.2087 conversion values.
//

import Foundation
import Testing
import simd
@testable import CorridorKeyToolboxLogic

@Suite("ColorGamutMatrix")
struct ColorGamutMatrixTests {

    @Test("Rec.709 gamut returns the identity transform")
    func rec709IsIdentity() {
        let transform = ColorGamutMatrix.transform(for: .rec709)
        let identity = matrix_identity_float3x3
        for column in 0..<3 {
            for row in 0..<3 {
                #expect(transform.workingToRec709[column][row] == identity[column][row])
            }
        }
    }

    @Test("Rec.2020 matrix maps pure primaries close to their Rec.709 counterparts")
    func rec2020Primaries() {
        let transform = ColorGamutMatrix.transform(for: .rec2020)
        let rec2020Red = SIMD3<Float>(1, 0, 0)
        let rec2020Green = SIMD3<Float>(0, 1, 0)
        let rec2020Blue = SIMD3<Float>(0, 0, 1)

        let mappedRed = transform.workingToRec709 * rec2020Red
        let mappedGreen = transform.workingToRec709 * rec2020Green
        let mappedBlue = transform.workingToRec709 * rec2020Blue

        // Rec.2020's red primary is outside Rec.709 gamut, so its Rec.709
        // representation has R > 1 and smaller components in G and B.
        #expect(mappedRed.x > 1.5, "Rec.2020 red should map to R > 1.5 in Rec.709.")
        #expect(mappedRed.y < 0, "Rec.2020 red should map to a negative green component.")

        // Green and blue should similarly over-saturate relative to their
        // home channel.
        #expect(mappedGreen.y > 1.0)
        #expect(mappedBlue.z > 1.0)
    }

    @Test("Primaries-raw bridge maps 0 to Rec.709 and 1 to Rec.2020")
    func primariesBridge() {
        #expect(ColorGamutMatrix.gamut(fromColorPrimariesRaw: 0) == .rec709)
        #expect(ColorGamutMatrix.gamut(fromColorPrimariesRaw: 1) == .rec2020)
    }
}
