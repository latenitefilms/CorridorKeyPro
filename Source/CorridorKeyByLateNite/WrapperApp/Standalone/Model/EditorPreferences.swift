//
//  EditorPreferences.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Stores the small set of inspector choices that should *persist
//  across editor sessions* — Quality, Hint, Upscale Method — so a
//  user who's tuned their workflow doesn't have to re-pick the same
//  three popups every time they open the editor.
//
//  Why these three specifically:
//  * Quality and Upscale Method express hardware preferences (a 16 GB
//    Mac mini wants Recommended + Lanczos forever; a 64 GB Studio
//    wants Maximum). They almost never need to vary per-clip.
//  * Hint reflects the user's chosen segmentation strategy (Apple
//    Vision vs. chroma vs. Manual), again typically a per-user choice
//    rather than a per-clip one.
//
//  Why NOT Screen Colour:
//  * Screen Colour is per-clip — a green-screen plate and a blue-screen
//    plate sit side-by-side in the same project. We auto-detect it
//    from the first frame on import (`ScreenColorAutoDetector`) and
//    let the user override per-clip; saving "last screen colour" as
//    a session default would actively fight that workflow.
//
//  Why NOT every parameter:
//  * Sliders (Black Point, Despill Strength, etc.) are per-clip
//    creative choices. The Reset to Default affordance handles
//    "factory" recovery; persisting them would conflict with that.
//
//  The keys are namespaced under the bundle identifier prefix used by
//  the rest of the app's `UserDefaults` writes so they show up
//  alongside FCP / window-frame state instead of polluting a separate
//  domain.
//

import Foundation

enum EditorPreferences {

    private enum Key {
        static let qualityMode = "ck.editor.qualityMode"
        static let hintMode = "ck.editor.hintMode"
        static let upscaleMethod = "ck.editor.upscaleMethod"
    }

    /// Reads the persisted Quality choice, falling back to the
    /// `ParameterRanges.Defaults` factory value when no entry exists
    /// (first launch) or the stored value is no longer a valid case
    /// (which would happen if a future build retired a Quality option
    /// that an older session had selected).
    static var qualityMode: QualityMode {
        get { decode(Key.qualityMode, default: ParameterRanges.Defaults.qualityMode) }
        set { encode(newValue, forKey: Key.qualityMode) }
    }

    static var hintMode: HintMode {
        get { decode(Key.hintMode, default: ParameterRanges.Defaults.hintMode) }
        set { encode(newValue, forKey: Key.hintMode) }
    }

    static var upscaleMethod: UpscaleMethod {
        get { decode(Key.upscaleMethod, default: ParameterRanges.Defaults.upscaleMethod) }
        set { encode(newValue, forKey: Key.upscaleMethod) }
    }

    /// Returns a `PluginStateData` that mixes the persisted preferences
    /// in on top of the regular factory defaults. The view model uses
    /// this as its starting state so a fresh editor window opens with
    /// the user's saved choices, while every other slider / toggle
    /// stays at its defaults.
    static func makeInitialState() -> PluginStateData {
        var state = PluginStateData()
        state.qualityMode = qualityMode
        state.hintMode = hintMode
        state.upscaleMethod = upscaleMethod
        return state
    }

    private static func decode<Value: RawRepresentable>(
        _ key: String,
        default fallback: Value
    ) -> Value where Value.RawValue == Int {
        guard UserDefaults.standard.object(forKey: key) != nil else {
            return fallback
        }
        let rawValue = UserDefaults.standard.integer(forKey: key)
        return Value(rawValue: rawValue) ?? fallback
    }

    private static func encode<Value: RawRepresentable>(
        _ value: Value,
        forKey key: String
    ) where Value.RawValue == Int {
        UserDefaults.standard.set(value.rawValue, forKey: key)
    }
}
