//
//  CorridorKeyProPlugIn+Render.swift
//  Corridor Key Toolbox
//
//  Hooks FxPlug's per-tile render callback into the Corridor Key render
//  pipeline. Final Cut Pro invokes this method on a background thread; the
//  render pipeline itself owns all Metal state internally.
//

import Foundation
import CoreMedia
import QuartzCore

extension CorridorKeyProPlugIn {

    @objc(renderDestinationImage:sourceImages:pluginState:atTime:error:)
    func renderDestinationImage(
        _ destinationImage: FxImageTile,
        sourceImages: [FxImageTile],
        pluginState: Data?,
        atTime renderTime: CMTime
    ) throws {
        guard let sourceImage = sourceImages.first else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_InvalidParameter,
                userInfo: [NSLocalizedDescriptionKey: "Corridor Key Toolbox requires a source input."]
            )
        }

        var state = PluginStateData.decoded(from: pluginState.map { NSData(data: $0) })
        let width = Int(destinationImage.imagePixelBounds.right - destinationImage.imagePixelBounds.left)
        let height = Int(destinationImage.imagePixelBounds.top - destinationImage.imagePixelBounds.bottom)
        state.destinationLongEdgePixels = max(width, height)

        let alphaHint = sourceImages.count > 1 ? sourceImages[1] : nil
        let request = RenderRequest(
            destinationImage: destinationImage,
            sourceImage: sourceImage,
            alphaHintImage: alphaHint,
            state: state,
            renderTime: renderTime
        )

        let startTime = CACurrentMediaTime()
        do {
            _ = try renderPipeline.render(request)
        } catch {
            PluginLog.error("Render failed at \(CMTimeGetSeconds(renderTime))s: \(error.localizedDescription)")
            throw error
        }
        let elapsedMilliseconds = (CACurrentMediaTime() - startTime) * 1000
        lastFrameMilliseconds.set(elapsedMilliseconds)
    }
}
