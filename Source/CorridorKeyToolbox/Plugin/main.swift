//
//  main.swift
//  Corridor Key Toolbox
//
//  Entry point for the FxPlug XPC service. `FxPrincipal.startServicePrincipal`
//  hands control over to Final Cut Pro's plug-in host, which instantiates
//  `CorridorKeyToolboxPlugIn` on demand. We install file-based logging before
//  handing off so a crash or early error is captured on disk.
//

import Foundation

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

let serviceDelegate = CorridorKeyToolboxServiceDelegate()
FxPrincipal.startServicePrincipal(with: serviceDelegate)
