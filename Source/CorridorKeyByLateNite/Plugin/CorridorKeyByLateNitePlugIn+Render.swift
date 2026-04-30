//
//  CorridorKeyToolboxPlugIn+Render.swift
//  CorridorKey by LateNite
//
//  Hooks FxPlug's per-tile render callback into the Corridor Key render
//  pipeline. Final Cut Pro invokes this method on a background thread; the
//  render pipeline itself owns all Metal state internally.
//

import Foundation
import CoreMedia
import QuartzCore

extension CorridorKeyToolboxPlugIn {

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
                userInfo: [NSLocalizedDescriptionKey: "CorridorKey by LateNite requires a source input."]
            )
        }

        var state = PluginStateData.decoded(from: pluginState.map { NSData(data: $0) })
        let width = Int(destinationImage.imagePixelBounds.right - destinationImage.imagePixelBounds.left)
        let height = Int(destinationImage.imagePixelBounds.top - destinationImage.imagePixelBounds.bottom)
        state.destinationLongEdgePixels = max(width, height)
        refreshCachedMatteIfAvailable(for: &state, at: renderTime)

        let gamut = currentWorkingGamut()
        let alphaHint = sourceImages.count > 1 ? sourceImages[1] : nil
        let request = RenderRequest(
            destinationImage: destinationImage,
            sourceImage: sourceImage,
            alphaHintImage: alphaHint,
            state: state,
            workingGamut: gamut,
            renderTime: renderTime
        )

        let startTime = CACurrentMediaTime()
        do {
            let report = try renderPipeline.render(request)
            lastRenderReport.set(report)
        } catch {
            PluginLog.error("Render failed at \(CMTimeGetSeconds(renderTime))s: \(error.localizedDescription)")
            throw error
        }
        let elapsedMilliseconds = (CACurrentMediaTime() - startTime) * 1000
        lastFrameMilliseconds.set(elapsedMilliseconds)
    }

    /// Queries the FxPlug colour gamut API. Unknown / missing → Rec.709 so
    /// the identity transform is used and behaviour matches the pre-v1.0
    /// implementation for SDR timelines.
    private func currentWorkingGamut() -> WorkingColorGamut {
        guard let gamutAPI = apiManager.api(for: (any FxColorGamutAPI_v2).self) as? any FxColorGamutAPI_v2 else {
            return .rec709
        }
        let raw = UInt(gamutAPI.colorPrimaries())
        return ColorGamutMatrix.gamut(fromColorPrimariesRaw: raw)
    }

    /// Rehydrates the per-frame matte from the in-memory analysis snapshot.
    ///
    /// FxPlug supplies `pluginState` as a host-managed blob. During frame
    /// stepping we can receive a blob generated before the hidden analysis
    /// parameter became visible for the requested time. The analysis snapshot
    /// store is updated whenever we successfully persist or load a cache, so
    /// prefer its current-frame lookup over the potentially stale embedded
    /// blob. If this render instance has not seen the cache yet, load it
    /// once from the hidden parameter and seed the same-process store.
    private func refreshCachedMatteIfAvailable(
        for state: inout PluginStateData,
        at renderTime: CMTime
    ) {
        if let cachedMatte = analysisSnapshotStore.cachedMatte(
            at: renderTime,
            screenColorRaw: state.screenColor.rawValue
        ) {
            state.cachedMatteBlob = cachedMatte.blob
            state.cachedMatteInferenceResolution = cachedMatte.inferenceResolution
            return
        }

        guard let retrieval = apiManager.api(
            for: (any FxParameterRetrievalAPI_v6).self
        ) as? any FxParameterRetrievalAPI_v6,
              let analysis = loadAnalysisData(using: retrieval),
              let cachedMatte = analysis.cachedMatte(
                  at: renderTime,
                  screenColorRaw: state.screenColor.rawValue
              )
        else { return }
        state.cachedMatteBlob = cachedMatte.blob
        state.cachedMatteInferenceResolution = cachedMatte.inferenceResolution
    }
}
