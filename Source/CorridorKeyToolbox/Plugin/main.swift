//
//  main.swift
//  Corridor Key Toolbox
//
//  Entry point for the FxPlug XPC service. `FxPrincipal.startServicePrincipal`
//  hands control over to Final Cut Pro's plug-in host, which instantiates
//  `CorridorKeyToolboxPlugIn` on demand. We install file-based logging before
//  handing off so a crash or early error is captured on disk.
//
//  Warm-up timing: MLX model loading + JIT takes ~2–5s. Kicking it off
//  here — *before* `FxPrincipal.startServicePrincipal` blocks — gives
//  the warm-up a head start of several hundred ms over the previous
//  pattern (kicking off in `CorridorKeyToolboxPlugIn.init`, which
//  fires only when FCP creates the first plug-in instance). On a
//  typical session the user takes a few seconds to add the effect
//  and click Analyse; that gap now overlaps with warm-up so the
//  first analyse frame doesn't pay the load+compile cost.
//

import Foundation
import Metal

/// Delegate that captures connection info from Final Cut Pro. FxPlug passes
/// the host bundle identifier and version to the renderer once the XPC channel
/// is alive; recording it makes diagnosing Motion / Final Cut Pro specific
/// issues a single log grep away.
final class CorridorKeyToolboxServiceDelegate: NSObject, FxPrincipalDelegate {
    func didEstablishConnection(withHost hostBundleIdentifier: String, version hostVersionString: String) {
        PluginLog.notice("FxPlug connected to host \(hostBundleIdentifier) (\(hostVersionString))")
    }
}

PluginLog.installFileRedirect()
PluginLog.notice("Corridor Key Toolbox Renderer launching.")

// Kick off MLX warm-up immediately at XPC service startup. The warm-up
// runs on a background utility task, so it doesn't block the principal
// from starting; it just gets a head start on loading the .mlxfn file
// and JIT-compiling the graph while FCP wires up the FxPlug interface.
// `SharedMLXBridgeRegistry.beginWarmup` is idempotent — the per-instance
// `kickOffDefaultWarmup` call in `CorridorKeyToolboxPlugIn.init` will
// hit the warm engine instead of starting a second pass.
if let device = MTLCreateSystemDefaultDevice(),
   let entry = try? MetalDeviceCache.shared.entry(for: device) {
    let defaultRung = QualityMode.automatic.resolvedInferenceResolution(
        forLongEdge: 1920,
        deviceRegistryID: device.registryID
    )
    PluginLog.notice("Pre-warming MLX bridge at \(defaultRung)px on \(device.name) before FCP attaches.")
    SharedMLXBridgeRegistry.shared.beginWarmup(
        deviceRegistryID: device.registryID,
        rung: defaultRung,
        cacheEntry: entry
    )
}

let serviceDelegate = CorridorKeyToolboxServiceDelegate()
FxPrincipal.startServicePrincipal(with: serviceDelegate)
