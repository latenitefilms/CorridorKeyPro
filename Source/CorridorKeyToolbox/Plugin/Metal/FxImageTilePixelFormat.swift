//
//  FxImageTilePixelFormat.swift
//  CorridorKey by LateNite
//
//  FxPlug-specific extension point split out so `MetalDeviceCache.swift`
//  stays free of FxPlug types and can be compiled by the Swift Package
//  (which runs the headless Metal tests).
//

import Foundation
import Metal
import CoreVideo

extension MetalDeviceCache {
    /// Maps the IOSurface pixel format FxPlug provides to the nearest Metal
    /// pixel format. Modern Final Cut Pro hands us 16-bit half floats for
    /// colour-managed output and BGRA8 for quick draft renders.
    static func metalPixelFormat(for tile: FxImageTile) -> MTLPixelFormat {
        switch tile.ioSurface.pixelFormat {
        case kCVPixelFormatType_128RGBAFloat: return .rgba32Float
        case kCVPixelFormatType_64RGBAHalf: return .rgba16Float
        case kCVPixelFormatType_32BGRA: return .bgra8Unorm
        default:
            PluginLog.warning(
                "Unexpected IOSurface pixel format \(String(tile.ioSurface.pixelFormat, radix: 16)); defaulting to rgba16Float."
            )
            return .rgba16Float
        }
    }
}
