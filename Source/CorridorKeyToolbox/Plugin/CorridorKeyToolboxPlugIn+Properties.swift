//
//  CorridorKeyToolboxPlugIn+Properties.swift
//  CorridorKey by LateNite
//
//  Declares the static capabilities of the plug-in. Properties are fetched
//  once per instance and cached by the host, so this method is the right place
//  to advertise things like colour-space preferences and tiling support.
//

import Foundation

extension CorridorKeyToolboxPlugIn {

    @objc(properties:error:)
    func properties(_ properties: AutoreleasingUnsafeMutablePointer<NSDictionary>?) throws {
        // ImageNet-trained networks expect gamma-encoded inputs (sRGB-ish), so
        // ask Final Cut Pro for video-gamma pixels. We also take the whole
        // image each render because the matte depends on spatial context
        // outside any individual tile.
        let swiftProperties: [String: Any] = [
            kFxPropertyKey_MayRemapTime: NSNumber(value: false),
            kFxPropertyKey_PixelTransformSupport: NSNumber(value: kFxPixelTransform_Full),
            kFxPropertyKey_VariesWhenParamsAreStatic: NSNumber(value: false),
            kFxPropertyKey_ChangesOutputSize: NSNumber(value: false),
            kFxPropertyKey_DesiredProcessingColorInfo: NSNumber(value: kFxImageColorInfo_RGB_GAMMA_VIDEO),
            kFxPropertyKey_NeedsFullBuffer: NSNumber(value: true)
        ]
        properties?.pointee = swiftProperties as NSDictionary
    }
}
