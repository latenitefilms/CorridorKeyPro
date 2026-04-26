//
//  HintPointSet.swift
//  CorridorKey by LateNite
//
//  Value type for the user-placed foreground / background hint points
//  the on-screen control writes into a custom parameter and the
//  renderer rasterises into the MLX bridge's 4th input channel.
//
//  The set is stored inside the same FCP Library as the Analysis Data
//  blob, which is why it round-trips through `NSDictionary` (FxPlug
//  custom-parameter contract).
//

import Foundation
import CoreGraphics

/// Foreground vs background hint label. Foreground points pull the
/// matte toward 1.0 in their neighbourhood; background points pull it
/// toward 0.0. Stored as `Int` to round-trip through NSNumber cleanly.
public enum HintPointKind: Int, Sendable, Codable {
    case foreground = 0
    case background = 1
}

/// One user-placed hint point. Coordinates are in object-normalized
/// space (0…1 across the input image) so they remain valid when the
/// user changes the canvas size, scrubs around timecodes, or applies
/// a transform on top of CorridorKey by LateNite.
public struct HintPoint: Hashable, Sendable, Codable {
    public var x: Double
    public var y: Double
    public var kind: HintPointKind
    public var radiusNormalized: Double

    public init(x: Double, y: Double, kind: HintPointKind, radiusNormalized: Double = 0.04) {
        self.x = x
        self.y = y
        self.kind = kind
        self.radiusNormalized = radiusNormalized
    }
}

/// Serializable set of hint points the OSC writes into the FxPlug
/// "Subject Points" custom parameter and the renderer reads back at
/// analysis time. Codable so unit tests can round-trip without
/// FxPlug.
public struct HintPointSet: Hashable, Sendable, Codable {
    public var points: [HintPoint]

    public init(points: [HintPoint] = []) {
        self.points = points
    }

    public var isEmpty: Bool { points.isEmpty }

    /// Persists to the host as a binary plist so we can embed it
    /// inside a custom-parameter NSDictionary alongside other custom
    /// payloads. Plist binary keeps the encoded blob compact and
    /// avoids JSON's percent-escaping bloat for long sequences.
    public func encodedForHost() throws -> Data {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        return try encoder.encode(self)
    }

    public static func decoded(from data: Data?) -> HintPointSet {
        guard let data, !data.isEmpty else { return HintPointSet() }
        do {
            return try PropertyListDecoder().decode(HintPointSet.self, from: data)
        } catch {
            return HintPointSet()
        }
    }

    /// Round-trips through the FxPlug custom-parameter NSDictionary
    /// contract. The dictionary has one key, "blob", whose value is
    /// the binary plist payload above. Future revisions can layer in
    /// extra keys (e.g. a schema version) without breaking older
    /// builds, which simply ignore unknown keys.
    public func asParameterDictionary() -> NSDictionary {
        let dict = NSMutableDictionary()
        if let blob = try? encodedForHost() {
            dict["blob"] = blob as NSData
        }
        return dict
    }

    public static func fromParameterDictionary(_ dictionary: NSDictionary?) -> HintPointSet {
        guard let dictionary, let blob = dictionary["blob"] as? Data else {
            return HintPointSet()
        }
        return decoded(from: blob)
    }

    // MARK: - Editing helpers

    public mutating func add(_ point: HintPoint) {
        points.append(point)
    }

    /// Removes the point closest to `(x, y)` if any sit within
    /// `tolerance` (in object-normalized units). Returns `true` when
    /// a point was removed, so the caller can decide whether to
    /// trigger a re-render. The OSC uses this for shift-click delete
    /// and right-click delete.
    @discardableResult
    public mutating func removeNearest(toX x: Double, y: Double, tolerance: Double) -> Bool {
        guard !points.isEmpty else { return false }
        var nearestIndex: Int? = nil
        var nearestDistanceSquared = tolerance * tolerance
        for (index, point) in points.enumerated() {
            let dx = point.x - x
            let dy = point.y - y
            let distanceSquared = dx * dx + dy * dy
            if distanceSquared <= nearestDistanceSquared {
                nearestDistanceSquared = distanceSquared
                nearestIndex = index
            }
        }
        if let nearestIndex {
            points.remove(at: nearestIndex)
            return true
        }
        return false
    }

    public mutating func clear() {
        points.removeAll(keepingCapacity: false)
    }
}
