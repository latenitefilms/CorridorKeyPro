//
//  MetalStagesResourceBundleAccessor.swift
//  Corridor Key Toolbox
//
//  SPM-only helper that exposes the Metal-stages module's `Bundle.module`
//  so test targets can find `CorridorKeyShaders.metal` and the shared
//  shader-types header without needing to bundle their own copies. The
//  compile flag keeps this file out of the Xcode build entirely — the
//  FxPlug target doesn't need it.
//

#if CORRIDOR_KEY_SPM_MIRROR

import Foundation

public enum MetalStagesResourceBundleAccessor {
    /// Returns the resource bundle of the `CorridorKeyToolboxMetalStages`
    /// SPM target. Holds the compiled shader source + types header that
    /// inference and Metal-stages tests need to spin up a real
    /// `MetalDeviceCacheEntry`.
    public static func bundle() -> Bundle {
        Bundle.module
    }
}

#endif
