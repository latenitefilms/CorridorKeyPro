//
//  CorridorKeyToolboxPlugIn+RenderRects.swift
//  Corridor Key Toolbox
//
//  Tells FxPlug which parts of the input and output images we touch. The
//  keyer is an identity transform so the output bounds always equal the
//  input bounds. Because the neural matte considers spatial context, we
//  also ask for the full source image rather than a sub-tile.
//

import Foundation
import CoreMedia

extension CorridorKeyToolboxPlugIn {

    @objc(destinationImageRect:sourceImages:destinationImage:pluginState:atTime:error:)
    func destinationImageRect(
        _ destinationImageRect: UnsafeMutablePointer<FxRect>,
        sourceImages: [FxImageTile],
        destinationImage: FxImageTile,
        pluginState: Data?,
        atTime renderTime: CMTime
    ) throws {
        if let source = sourceImages.first {
            destinationImageRect.pointee = source.imagePixelBounds
        } else {
            destinationImageRect.pointee = destinationImage.imagePixelBounds
        }
    }

    @objc(sourceTileRect:sourceImageIndex:sourceImages:destinationTileRect:destinationImage:pluginState:atTime:error:)
    func sourceTileRect(
        _ sourceTileRect: UnsafeMutablePointer<FxRect>,
        sourceImageIndex: UInt,
        sourceImages: [FxImageTile],
        destinationTileRect: FxRect,
        destinationImage: FxImageTile,
        pluginState: Data?,
        atTime renderTime: CMTime
    ) throws {
        let index = Int(sourceImageIndex)
        if sourceImages.indices.contains(index) {
            sourceTileRect.pointee = sourceImages[index].imagePixelBounds
        } else {
            sourceTileRect.pointee = destinationTileRect
        }
    }
}
