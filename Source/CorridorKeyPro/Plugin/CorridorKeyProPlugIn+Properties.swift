//
//  CorridorKeyProPlugIn+Properties.swift
//  Corridor Key Pro
//
//  Declares the static capabilities of the plug-in. Properties are fetched
//  once per instance and cached by the host, so this method is the right place
//  to advertise things like colour-space preferences and tiling support.
//

import Foundation

extension CorridorKeyProPlugIn {

    @objc(properties:error:)
    func properties(_ properties: AutoreleasingUnsafeMutablePointer<NSDictionary>?) throws {
        // The matte network benefits from linear input; ask Final Cut Pro to
        // hand us linear Rec.709 whenever the project colour management allows
        // it. `kFxImageColorInfo_RGB_LINEAR` keeps the gamma-correct path
        // active across the entire GPU pipeline.
        let swiftProperties: [String: Any] = [
            kFxPropertyKey_MayRemapTime: NSNumber(value: false),
            kFxPropertyKey_PixelTransformSupport: NSNumber(value: kFxPixelTransform_ScaleTranslate),
            kFxPropertyKey_VariesWhenParamsAreStatic: NSNumber(value: false),
            kFxPropertyKey_ChangesOutputSize: NSNumber(value: false),
            kFxPropertyKey_DesiredProcessingColorInfo: NSNumber(value: kFxImageColorInfo_RGB_LINEAR),
            kFxPropertyKey_NeedsFullBuffer: NSNumber(value: false)
        ]
        properties?.pointee = swiftProperties as NSDictionary
    }
}
