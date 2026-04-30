//
//  HintPointSetTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Round-trip and editing semantics for the on-screen control's
//  hint-point storage. The set is the single source of truth that
//  travels through the FCP Library, so any encoding regression or
//  off-by-one in `removeNearest` would lose user-placed dots.
//

import Foundation
import Testing
@testable import CorridorKeyToolboxLogic

@Suite("HintPointSet")
struct HintPointSetTests {

    @Test("Round-trip via parameter dictionary preserves every point")
    func dictionaryRoundTrip() throws {
        let original = HintPointSet(points: [
            HintPoint(x: 0.25, y: 0.5, kind: .foreground, radiusNormalized: 0.05),
            HintPoint(x: 0.75, y: 0.5, kind: .background, radiusNormalized: 0.04)
        ])
        let dict = original.asParameterDictionary()
        let decoded = HintPointSet.fromParameterDictionary(dict)
        #expect(decoded.points.count == 2)
        #expect(decoded.points[0].kind == .foreground)
        #expect(decoded.points[0].x == 0.25)
        #expect(decoded.points[1].kind == .background)
        #expect(decoded.points[1].radiusNormalized == 0.04)
    }

    @Test("Dictionary decode accepts NSString keys and NSData values")
    func nsDictionaryRoundTrip() throws {
        let original = HintPointSet(points: [
            HintPoint(x: 0.45, y: 0.55, kind: .foreground, radiusNormalized: 0.03)
        ])
        let dictionary = NSMutableDictionary()
        dictionary[NSString(string: "blob")] = try original.encodedForHost() as NSData

        let decoded = HintPointSet.fromParameterDictionary(dictionary)

        #expect(decoded.points.count == 1)
        #expect(decoded.points[0].x == 0.45)
        #expect(decoded.points[0].kind == .foreground)
    }

    @Test("Empty dictionary decodes to empty set")
    func emptyDictionary() {
        let decoded = HintPointSet.fromParameterDictionary(NSDictionary())
        #expect(decoded.isEmpty)
    }

    @Test("nil dictionary decodes to empty set")
    func nilDictionary() {
        let decoded = HintPointSet.fromParameterDictionary(nil)
        #expect(decoded.isEmpty)
    }

    @Test("removeNearest removes the closest point inside tolerance")
    func removeNearestInsideTolerance() {
        var set = HintPointSet(points: [
            HintPoint(x: 0.1, y: 0.1, kind: .foreground),
            HintPoint(x: 0.5, y: 0.5, kind: .foreground),
            HintPoint(x: 0.9, y: 0.9, kind: .foreground)
        ])
        let removed = set.removeNearest(toX: 0.51, y: 0.49, tolerance: 0.05)
        #expect(removed == true)
        #expect(set.points.count == 2)
        #expect(set.points[0].x == 0.1)
        #expect(set.points[1].x == 0.9)
    }

    @Test("removeNearest leaves points outside tolerance alone")
    func removeNearestOutsideTolerance() {
        var set = HintPointSet(points: [
            HintPoint(x: 0.1, y: 0.1, kind: .foreground)
        ])
        let removed = set.removeNearest(toX: 0.5, y: 0.5, tolerance: 0.05)
        #expect(removed == false)
        #expect(set.points.count == 1)
    }

    @Test("clear empties the set")
    func clearEmpties() {
        var set = HintPointSet(points: [
            HintPoint(x: 0.1, y: 0.1, kind: .foreground),
            HintPoint(x: 0.9, y: 0.9, kind: .background)
        ])
        set.clear()
        #expect(set.isEmpty)
    }
}
