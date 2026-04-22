//
//  CorridorKeyProPlugIn+RenderRects.swift
//  Corridor Key Pro
//
//  Tells FxPlug which parts of the input and output images we touch. The
//  keyer is an identity transform — the output bounds always match the input
//  bounds — and we need to read the full source tile to produce a given
//  output tile because the neural matte considers spatial context.
//

import Foundation
import CoreMedia

extension CorridorKeyProPlugIn {

    @objc(destinationImageRect:sourceImages:destinationImage:pluginState:atTime:error:)
    func destinationImageRect(
        _ destinationImageRect: UnsafeMutablePointer<FxRect>,
        sourceImages: [FxImageTile],
        destinationImage: FxImageTile,
        pluginState: Data?,
        at renderTime: CMTime
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
        at renderTime: CMTime
    ) throws {
        // We need the complete source image to produce a spatially consistent
        // matte, so expand the required input to the full image bounds.
        let index = Int(sourceImageIndex)
        if sourceImages.indices.contains(index) {
            sourceTileRect.pointee = sourceImages[index].imagePixelBounds
        } else {
            sourceTileRect.pointee = destinationTileRect
        }
    }
}
