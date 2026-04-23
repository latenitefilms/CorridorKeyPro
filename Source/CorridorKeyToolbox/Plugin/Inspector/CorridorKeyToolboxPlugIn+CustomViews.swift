//
//  CorridorKeyToolboxPlugIn+CustomViews.swift
//  Corridor Key Toolbox
//
//  Hands Final Cut Pro the SwiftUI-backed inspector header for the header
//  custom-UI parameter. FxPlug takes ownership of the returned NSView via
//  `Unmanaged.passRetained` so we don't autorelease it out from under the
//  host.
//

import AppKit
import Foundation
import SwiftUI

/// NSHostingView subclass with two jobs: stop the header from stealing
/// first-responder when users click into the inspector, and own a strong
/// reference to the `CorridorKeyInspectorBridge` so the SwiftUI ObservedObject
/// subscription can't outlive its publisher if Final Cut Pro retains the
/// view longer than we expect.
@MainActor
private final class CorridorKeyHostingView<Content: View>: NSHostingView<Content>, @unchecked Sendable {
    /// Held so the bridge's lifetime is pinned to the hosting view FxPlug
    /// retains — without this, SwiftUI's view struct was the only thing
    /// keeping the bridge alive, and intermittent re-renders could drop it
    /// and leave the header blank.
    var retainedBridge: CorridorKeyInspectorBridge?

    override var acceptsFirstResponder: Bool { false }
    override func becomeFirstResponder() -> Bool { false }
}

extension CorridorKeyToolboxPlugIn {

    @objc(createViewForParameterID:)
    func createViewForParameter(_ parameterID: UInt32) -> NSView? {
        guard parameterID == ParameterIdentifier.headerSummary else {
            return nil
        }

        nonisolated(unsafe) let apiManagerRef = apiManager
        nonisolated(unsafe) let pluginRef = self

        let hostingView: CorridorKeyHostingView<CorridorKeyHeaderView> = MainActor.assumeIsolated {
            let bridge = CorridorKeyInspectorBridge(apiManager: apiManagerRef, plugin: pluginRef)
            let view = CorridorKeyHostingView(rootView: CorridorKeyHeaderView(bridge: bridge))
            view.retainedBridge = bridge
            view.frame = NSRect(x: 0, y: 0, width: 320, height: 150)
            view.autoresizingMask = [.width]
            return view
        }

        return Unmanaged.passRetained(hostingView).takeUnretainedValue()
    }
}
