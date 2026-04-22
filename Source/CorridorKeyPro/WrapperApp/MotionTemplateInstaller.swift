//
//  MotionTemplateInstaller.swift
//  Corridor Key Pro
//
//  Installs (or refreshes) the bundled Motion Template into the user's
//  `~/Movies/Motion Templates.localized/Effects.localized/Corridor Key Pro/`
//  directory so Final Cut Pro picks up the effect as soon as the app finishes
//  launching. The installer is idempotent — re-running it with an unchanged
//  bundle is a no-op, and re-running after an upgrade replaces the previous
//  copy on disk.
//

import Foundation
import AppKit

// MARK: - Result types

/// Terminal state surfaced back to the SwiftUI view once installation
/// finishes. Two success cases keep the UI messaging meaningful without
/// exposing disk-level details.
enum MotionTemplateInstallationResult: Sendable {
    case alreadyInstalled
    case installed
    case failed(title: String, info: String)
}

/// Internal result used while copying files. Kept private so the SwiftUI
/// layer only has to reason about the higher-level `MotionTemplateInstallationResult`.
private enum MotionTemplateCopyResult: Sendable {
    case success
    case failed(title: String, info: String)
}

// MARK: - Installer

/// Performs the Motion Template install work off of the main actor so the
/// SwiftUI view stays responsive even when the file operations take a moment
/// (for example, replacing a large media folder on a slow disk).
actor MotionTemplateInstaller {
    static let effectCategory = "Corridor Key Pro"

    private let fileManager = FileManager.default

    /// Entry point: installs (or updates) the latest bundled template and
    /// returns a `MotionTemplateInstallationResult` describing the outcome.
    func installLatestTemplate() -> MotionTemplateInstallationResult {
        guard let bundledTemplateURL = bundledTemplateURL() else {
            return .failed(
                title: "Motion Template could not be installed.",
                info: "Motion Template resources were not found inside Corridor Key Pro.app."
            )
        }

        let moviesFolderURL = URL.moviesDirectory
        if isMotionTemplateAlreadyInstalled(in: moviesFolderURL, bundledTemplateURL: bundledTemplateURL) {
            return .alreadyInstalled
        }

        switch installMotionTemplate(in: moviesFolderURL, bundledTemplateURL: bundledTemplateURL) {
        case .success:
            break
        case .failed(let title, let info):
            return .failed(title: title, info: info)
        }

        guard isMotionTemplateAlreadyInstalled(in: moviesFolderURL, bundledTemplateURL: bundledTemplateURL) else {
            return .failed(
                title: "Motion Template could not be verified.",
                info: "Corridor Key Pro copied the template files but could not confirm the installed copy. Try relaunching the app."
            )
        }

        return .installed
    }

    // MARK: - Paths

    /// Location of the bundled template inside the running wrapper app. The
    /// Motion Template folder reference in the Xcode project lands at
    /// `Contents/Resources/Motion Template/Corridor Key Pro`.
    private func bundledTemplateURL() -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else {
            return nil
        }
        return resourceURL
            .appending(path: "Motion Template", directoryHint: .isDirectory)
            .appending(path: Self.effectCategory, directoryHint: .isDirectory)
    }

    /// Final Cut Pro expects effects at:
    /// `~/Movies/Motion Templates.localized/Effects.localized/<Category>/<TemplateName>/`
    /// We use the same name for the category and the template so a single
    /// effect appears in the inspector browser.
    private func destinationTemplateURL(in moviesFolderURL: URL) -> URL {
        moviesFolderURL
            .appending(path: "Motion Templates.localized", directoryHint: .isDirectory)
            .appending(path: "Effects.localized", directoryHint: .isDirectory)
            .appending(path: Self.effectCategory, directoryHint: .isDirectory)
            .appending(path: Self.effectCategory, directoryHint: .isDirectory)
    }

    // MARK: - Freshness check

    /// Compares the bundled and installed templates byte-for-byte via
    /// `FileManager.contentsEqual`. Motion Template folders are relatively
    /// small so this is quick even on spinning disks.
    private func isMotionTemplateAlreadyInstalled(
        in moviesFolderURL: URL,
        bundledTemplateURL: URL
    ) -> Bool {
        let destinationURL = destinationTemplateURL(in: moviesFolderURL)
        guard fileManager.fileExists(atPath: destinationURL.path) else {
            log("No installed template at: \(destinationURL.path)")
            return false
        }

        if fileManager.contentsEqual(
            atPath: bundledTemplateURL.path,
            andPath: destinationURL.path
        ) {
            return true
        }

        log("Installed template differs from bundled version; will refresh.")
        return false
    }

    // MARK: - Copy

    /// Walks the required `~/Movies/...` folder chain, ensuring each level
    /// exists and is writable before copying the bundled template in.
    private func installMotionTemplate(
        in moviesFolderURL: URL,
        bundledTemplateURL: URL
    ) -> MotionTemplateCopyResult {
        switch ensureDirectoryExistsAndIsWritable(
            at: moviesFolderURL,
            pathDescription: "~/Movies"
        ) {
        case .success: break
        case .failed(let title, let info): return .failed(title: title, info: info)
        }

        let motionTemplatesURL = moviesFolderURL
            .appending(path: "Motion Templates.localized", directoryHint: .isDirectory)
        switch ensureDirectoryExistsAndIsWritable(
            at: motionTemplatesURL,
            pathDescription: "~/Movies/Motion Templates.localized"
        ) {
        case .success: break
        case .failed(let title, let info): return .failed(title: title, info: info)
        }

        let effectsURL = motionTemplatesURL
            .appending(path: "Effects.localized", directoryHint: .isDirectory)
        switch ensureDirectoryExistsAndIsWritable(
            at: effectsURL,
            pathDescription: "~/Movies/Motion Templates.localized/Effects.localized"
        ) {
        case .success: break
        case .failed(let title, let info): return .failed(title: title, info: info)
        }

        // Replace the effect category folder wholesale so a previous version
        // cannot leave stale files behind.
        let templateCategoryURL = effectsURL
            .appending(path: Self.effectCategory, directoryHint: .isDirectory)

        if fileManager.fileExists(atPath: templateCategoryURL.path) {
            do {
                try fileManager.removeItem(at: templateCategoryURL)
                log("Removed existing template folder at '\(templateCategoryURL.path)'.")
            } catch {
                return .failed(
                    title: "Motion Template could not be updated.",
                    info: error.localizedDescription
                )
            }
        }

        switch ensureDirectoryExistsAndIsWritable(
            at: templateCategoryURL,
            pathDescription: "~/Movies/Motion Templates.localized/Effects.localized/\(Self.effectCategory)"
        ) {
        case .success: break
        case .failed(let title, let info): return .failed(title: title, info: info)
        }

        let destinationURL = templateCategoryURL
            .appending(path: Self.effectCategory, directoryHint: .isDirectory)
        do {
            log("Copying template from '\(bundledTemplateURL.path)' to '\(destinationURL.path)'.")
            try fileManager.copyItem(at: bundledTemplateURL, to: destinationURL)
        } catch {
            return .failed(
                title: "Motion Template could not be installed.",
                info: error.localizedDescription
            )
        }
        return .success
    }

    /// Creates the directory if missing, then confirms it is indeed a
    /// writable folder. Keeps the copy step free of cascading `if` checks.
    private func ensureDirectoryExistsAndIsWritable(
        at directoryURL: URL,
        pathDescription: String
    ) -> MotionTemplateCopyResult {
        if !fileManager.fileExists(atPath: directoryURL.path) {
            do {
                try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            } catch {
                return .failed(
                    title: "Motion Template could not be installed.",
                    info: "The '\(pathDescription)' folder could not be created."
                )
            }
        }

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directoryURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return .failed(
                title: "Motion Template could not be installed.",
                info: "The '\(pathDescription)' path is not a folder."
            )
        }

        guard fileManager.isWritableFile(atPath: directoryURL.path) else {
            return .failed(
                title: "Motion Template could not be installed.",
                info: "The '\(pathDescription)' folder is not writable."
            )
        }

        return .success
    }

    // MARK: - Logging

    private func log(_ message: String) {
        NSLog("[Corridor Key Pro] %@", message)
    }
}
