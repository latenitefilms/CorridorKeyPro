//
//  CorridorKeyProPlugIn+Render.swift
//  Corridor Key Pro
//
//  Hooks FxPlug's per-tile render callback into the Corridor Key render
//  pipeline. Final Cut Pro calls this method on a worker thread; we marshal to
//  the main actor because `RenderPipeline` owns per-instance Metal state.
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
        at renderTime: CMTime
    ) throws {
        guard let sourceImage = sourceImages.first else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_InvalidParameter,
                userInfo: [NSLocalizedDescriptionKey: "Corridor Key Pro requires a source input."]
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
        try renderPipeline.render(request)
        let elapsedMs = (CACurrentMediaTime() - startTime) * 1000
        lastFrameMilliseconds.set(elapsedMs)

        publishRuntimeStatus(for: state, elapsedMilliseconds: elapsedMs)
    }

    /// Updates the read-only status parameters so the user can see which
    /// backend is active and how fast the frame ran. The setting API works
    /// regardless of the calling thread; FxPlug schedules the write back onto
    /// the main UI queue internally.
    private func publishRuntimeStatus(for state: PluginStateData, elapsedMilliseconds: Double) {
        guard let setting = apiManager.api(for: FxParameterSettingAPI_v6.self) as? FxParameterSettingAPI_v6 else {
            return
        }
        let effectiveResolution = state.qualityMode.resolvedInferenceResolution(
            forLongEdge: state.destinationLongEdgePixels
        )
        setting.setStringParameterValue("\(effectiveResolution)px", toParameter: ParameterIdentifier.statusEffectiveQuality)

        let millisecondsText = String(
            format: "%.1f ms",
            locale: Locale(identifier: "en_US_POSIX"),
            elapsedMilliseconds
        )
        setting.setStringParameterValue(millisecondsText, toParameter: ParameterIdentifier.statusLastFrameMs)

        // Backend and device descriptions are filled in by the coordinator on
        // the first frame; the plugin pipeline is the owner of that state so
        // the render path does not need a fresh query here.
    }
}
