//
//  WelcomeView.swift
//  CorridorKey by LateNite
//
//  First-run welcome screen. Automatically installs or refreshes the bundled
//  Motion Template into the user's `~/Movies/Motion Templates.localized/...`
//  folder so Final Cut Pro picks up the effect the moment the wrapper launches.
//  The two buttons let the user jump straight into Final Cut Pro or open the
//  project's help page.
//

import SwiftUI
import AppKit

/// Coarse-grained state surfaced next to the install log so the user can see
/// at a glance whether the template is ready to use in Final Cut Pro.
private enum MotionTemplateInstallationState {
    case checking
    case installed
    case failed
}

struct WelcomeView: View {
    @Environment(\.openWindow) private var openWindow
    @State private var installationState: MotionTemplateInstallationState = .checking
    @State private var hasStartedInstall = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    @State private var isShowingAlert = false

    var body: some View {
        VStack(spacing: 22) {
            Spacer(minLength: 4)
            Image(nsImage: NSApplication.shared.applicationIconImage)
                .resizable()
                .interpolation(.high)
                .antialiased(true)
                .scaledToFit()
                .frame(width: 96, height: 96)
                .clipShape(.rect(cornerRadius: 20))
                .shadow(radius: 2, y: 1)

            Text("CorridorKey by LateNite")
                .font(.largeTitle)
                .bold()

            Text("Use the standalone editor to key any clip — or jump straight to \(Text("Final Cut Pro").bold()) and find CorridorKey by LateNite in the \(Text("Effects").bold()) Browser.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 40)

            InstallationStatusLabel(state: installationState)

            VStack(spacing: 14) {
                HStack(spacing: 14) {
                    Button("Open Standalone Editor", systemImage: "wand.and.stars") {
                        openWindow(id: EditorWindow.id)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                    Button("Open Final Cut Pro", systemImage: "play.rectangle") {
                        openFinalCutPro()
                    }
                    .controlSize(.large)
                }

                HStack(spacing: 14) {
                    Button("User Guide", systemImage: "book") {
                        openUserGuide()
                    }
                    .controlSize(.large)

                    Button("Reveal Log Files", systemImage: "doc.text.magnifyingglass") {
                        revealRendererLogs()
                    }
                    .controlSize(.large)
                }
            }
            .padding(.top, 4)
            Spacer(minLength: 4)
        }
        .padding()
        .onAppear {
            startAutomaticInstallIfNeeded()
        }
        .alert(alertTitle, isPresented: $isShowingAlert, actions: {
            Button("OK", role: .cancel) { }
        }, message: {
            Text(alertMessage)
        })
    }

    /// Bundle identifiers we try when looking for an installed copy of Final
    /// Cut Pro. The current App Store release ships as `com.apple.FinalCut`,
    /// historic builds used `com.apple.FinalCutApp`, and the time-limited trial
    /// is `com.apple.FinalCutTrial` – we accept all three so users on any
    /// path can launch straight into the host.
    private static let finalCutBundleIdentifiers = [
        "com.apple.FinalCut",
        "com.apple.FinalCutApp",
        "com.apple.FinalCutTrial"
    ]

    private func openFinalCutPro() {
        for bundleIdentifier in Self.finalCutBundleIdentifiers {
            if let finalCutURL = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleIdentifier) {
                NSWorkspace.shared.open(finalCutURL)
                return
            }
        }

        showAlert(
            title: "Final Cut Pro Not Found",
            message: "Install Final Cut Pro or Final Cut Pro Trial, then try again."
        )
    }

    private func openUserGuide() {
        guard let url = URL(string: "https://corridorkeytoolbox.fcp.cafe") else { return }
        NSWorkspace.shared.open(url)
    }

    /// Opens Finder at the renderer's log directory. The FxPlug renderer runs
    /// in its own sandbox container, so logs land at
    /// `~/Library/Containers/com.latenitefilms.CorridorKeyToolbox.Renderer/
    ///     Data/Library/Application Support/CorridorKey by LateNite/Logs`.
    /// NSWorkspace can reveal paths across sandbox boundaries because Finder
    /// performs the navigation in its own process.
    private func revealRendererLogs() {
        let rendererContainerURL = URL(fileURLWithPath: realUserHomePath())
            .appending(path: "Library/Containers/com.latenitefilms.CorridorKeyToolbox.Renderer/Data/Library/Application Support/CorridorKey by LateNite/Logs", directoryHint: .isDirectory)

        // If the directory doesn't exist yet (FCP hasn't launched the
        // renderer) let the user know rather than opening Finder at a
        // missing path.
        if !FileManager.default.fileExists(atPath: rendererContainerURL.path) {
            showAlert(
                title: "No logs yet",
                message: "The renderer log folder appears once Final Cut Pro has loaded CorridorKey by LateNite at least once. Apply the effect to a clip and try again."
            )
            return
        }

        NSWorkspace.shared.activateFileViewerSelecting([rendererContainerURL])
    }

    /// Resolves the real user home directory, sidestepping the wrapper app's
    /// sandbox container redirect. Required because the renderer's container
    /// lives under the real `~/Library/Containers`, not the wrapper app's
    /// redirected library.
    private func realUserHomePath() -> String {
        guard let passwordEntry = getpwuid(getuid()),
              let homeCString = passwordEntry.pointee.pw_dir else {
            return NSHomeDirectory()
        }
        return String(cString: homeCString)
    }

    /// Kicks off the Motion Template installation exactly once per view
    /// lifetime. `onAppear` may fire more than once during a session so we
    /// guard with a flag.
    private func startAutomaticInstallIfNeeded() {
        guard !hasStartedInstall else { return }
        hasStartedInstall = true
        installationState = .checking

        Task {
            let installer = MotionTemplateInstaller()
            let result = await installer.installLatestTemplate()
            await MainActor.run {
                applyInstallationResult(result)
            }
        }
    }

    @MainActor
    private func applyInstallationResult(_ result: MotionTemplateInstallationResult) {
        switch result {
        case .alreadyInstalled, .installed:
            installationState = .installed
        case .failed(let title, let info):
            installationState = .failed
            showAlert(title: title, message: info)
        }
    }

    private func showAlert(title: String, message: String) {
        alertTitle = title
        alertMessage = message
        isShowingAlert = true
    }
}

/// Compact status label shown under the welcome copy. Keeping it as its own
/// `View` struct avoids any `@ViewBuilder` computed property on `WelcomeView`.
private struct InstallationStatusLabel: View {
    let state: MotionTemplateInstallationState

    var body: some View {
        switch state {
        case .checking:
            Label("Installing Motion Template…", systemImage: "clock")
                .foregroundStyle(.orange)
        case .installed:
            Label("Motion Template installed and up to date.", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed:
            Label("Motion Template installation failed.", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        }
    }
}
