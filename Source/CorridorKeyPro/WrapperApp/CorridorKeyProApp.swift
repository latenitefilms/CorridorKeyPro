//
//  CorridorKeyProApp.swift
//  Corridor Key Pro
//
//  The wrapper application is required by Apple's FxPlug packaging model: the
//  FxPlug plug-in ships as an XPC service embedded inside a regular macOS
//  .app, so double-clicking the wrapper reveals the bundled service to Final
//  Cut Pro. We use the opportunity to greet the user, show the version, and
//  provide a single primary action for opening the installed folder.
//

import SwiftUI
import AppKit

@main
struct CorridorKeyProApp: App {
    var body: some Scene {
        WindowGroup("Corridor Key Pro") {
            WelcomeView()
                .frame(minWidth: 550, minHeight: 400)
                .frame(maxWidth: 550, maxHeight: 400)
                .preferredColorScheme(.dark)
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Corridor Key Pro") {
                    NSApp.orderFrontStandardAboutPanel(nil)
                }
            }
        }
    }
}
