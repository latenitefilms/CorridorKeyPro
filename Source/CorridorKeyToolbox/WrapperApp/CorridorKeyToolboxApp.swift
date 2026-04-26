//
//  CorridorKeyToolboxApp.swift
//  CorridorKey by LateNite
//
//  The wrapper application has two roles in v1.0:
//
//  1. Final Cut Pro plug-in host. Apple's FxPlug packaging model
//     requires the plug-in to ship as an XPC service embedded inside a
//     regular macOS .app — double-clicking the wrapper reveals the
//     bundled service to Final Cut Pro. The Welcome window installs
//     and refreshes the bundled Motion Template so the effect appears
//     in FCP's Effects browser the moment the wrapper launches.
//
//  2. Standalone editor. The same Corridor Key render pipeline that
//     powers the FxPlug also drives a built-in editor that imports a
//     clip via AV Foundation, previews it with the FCP parameters,
//     and exports the keyed result as Apple ProRes 4444. This makes
//     the wrapper genuinely useful for users who do not own Final
//     Cut Pro.
//
//  Both windows can be open simultaneously. The Welcome window is
//  the primary one (single-instance, app-style); the Editor window
//  uses the multi-window scene so power users can keep several
//  reference clips open side by side.
//

import SwiftUI
import AppKit

/// Stable identifier for the editor window so menu commands and the
/// Welcome screen open the same `WindowGroup` scene.
enum EditorWindow {
    static let id = "corridor-key-standalone-editor"
}

@main
struct CorridorKeyToolboxApp: App {
    var body: some Scene {
        WindowGroup("CorridorKey by LateNite") {
            WelcomeView()
                .frame(minWidth: 550, minHeight: 460)
                .frame(maxWidth: 550, maxHeight: 460)
                .preferredColorScheme(.dark)
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About CorridorKey by LateNite") {
                    NSApp.orderFrontStandardAboutPanel(nil)
                }
            }
            CommandGroup(replacing: .newItem) {
                NewEditorMenuItem()
            }
            CommandGroup(replacing: .help) {
                Button("CorridorKey by LateNite Help") {
                    if let url = URL(string: "https://corridorkeytoolbox.fcp.cafe") {
                        NSWorkspace.shared.open(url)
                    }
                }
            }
        }

        WindowGroup("Editor", id: EditorWindow.id) {
            EditorView()
                .preferredColorScheme(.dark)
                .frame(minWidth: 1100, minHeight: 700)
        }
        .windowResizability(.contentMinSize)
    }
}

/// Tiny view that pulls `openWindow` from the SwiftUI environment so a
/// command-bar item can spawn the editor window. Lives in its own
/// struct so the environment is wired through correctly when the
/// command appears in a non-view context.
private struct NewEditorMenuItem: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("New Editor Window") {
            openWindow(id: EditorWindow.id)
        }
        .keyboardShortcut("N", modifiers: [.command])
    }
}
