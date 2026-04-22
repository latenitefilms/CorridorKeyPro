//
//  CorridorKeyProPlugIn+CustomViews.swift
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

/// Prevents the header from stealing focus whenever the user clicks into
/// the inspector. Without this, pressing arrow keys after opening the
/// effect would scrub focus between subviews instead of reaching FCP.
@MainActor
private final class CorridorKeyHostingView<Content: View>: NSHostingView<Content>, @unchecked Sendable {
    override var acceptsFirstResponder: Bool { false }
    override func becomeFirstResponder() -> Bool { false }
}

extension CorridorKeyProPlugIn {

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
            view.frame = NSRect(x: 0, y: 0, width: 320, height: 150)
            view.autoresizingMask = [.width]
            return view
        }

        return Unmanaged.passRetained(hostingView).takeUnretainedValue()
    }
}
