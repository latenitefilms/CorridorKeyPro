//
//  MotionTemplateInstaller.swift
//  Corridor Key Pro
//
//  Installs (or refreshes) the bundled Motion Template into the user's
//  real `~/Movies/Motion Templates.localized/Effects.localized/Corridor Key
//  Pro/` directory so Final Cut Pro picks up the effect as soon as the app
//  finishes launching.
//
//  Because the wrapper app runs inside the App Sandbox,
//  `URL.moviesDirectory` resolves to the container's redirected copy at
//  `~/Library/Containers/com.latenitefilms.CorridorKeyPro/Data/Movies/` —
//  which Final Cut Pro never reads. We resolve the real user home with
//  `getpwuid`, which bypasses the sandbox redirect, and rely on the
//  `com.apple.security.assets.movies.read-write` entitlement to let the
//  copy succeed at the actual path.
//

import Foundation
import AppKit

// MARK: - Result types

enum MotionTemplateInstallationResult: Sendable {
    case alreadyInstalled
    case installed
    case failed(title: String, info: String)
}

private enum MotionTemplateCopyResult: Sendable {
    case success
    case failed(title: String, info: String)
}

// MARK: - Installer

actor MotionTemplateInstaller {
    static let effectCategory = "Corridor Key Pro"

    private let fileManager = FileManager.default

    /// Installs or refreshes the bundled Motion Template and returns a
    /// `MotionTemplateInstallationResult` describing the outcome.
    func installLatestTemplate() -> MotionTemplateInstallationResult {
        guard let bundledTemplateURL = bundledTemplateURL() else {
            return .failed(
                title: "Motion Template could not be installed.",
                info: "Motion Template resources were not found inside Corridor Key Pro.app."
            )
        }

        guard let realMoviesURL = realUserMoviesDirectory() else {
            return .failed(
                title: "Motion Template could not be installed.",
                info: "Corridor Key Pro could not locate your ~/Movies folder."
            )
        }

        if isMotionTemplateAlreadyInstalled(in: realMoviesURL, bundledTemplateURL: bundledTemplateURL) {
            return .alreadyInstalled
        }

        switch installMotionTemplate(in: realMoviesURL, bundledTemplateURL: bundledTemplateURL) {
        case .success:
            break
        case .failed(let title, let info):
            return .failed(title: title, info: info)
        }

        guard isMotionTemplateAlreadyInstalled(in: realMoviesURL, bundledTemplateURL: bundledTemplateURL) else {
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

    /// Resolves the **real** `~/Movies` path, ignoring the sandbox
    /// container redirect that `URL.moviesDirectory` and
    /// `FileManager.homeDirectoryForCurrentUser` apply. The POSIX password
    /// database returns the on-disk home directory set by the operating
    /// system, which is what Final Cut Pro reads its Motion Templates from.
    private func realUserMoviesDirectory() -> URL? {
        guard let passwordEntry = getpwuid(getuid()),
              let homeCString = passwordEntry.pointee.pw_dir else {
            return nil
        }
        let homePath = String(cString: homeCString)
        return URL(fileURLWithPath: homePath, isDirectory: true)
            .appending(path: "Movies", directoryHint: .isDirectory)
    }

    /// Final Cut Pro expects effects at:
    /// `~/Movies/Motion Templates.localized/Effects.localized/<Category>/<TemplateName>/`
    private func destinationTemplateURL(in moviesFolderURL: URL) -> URL {
        moviesFolderURL
            .appending(path: "Motion Templates.localized", directoryHint: .isDirectory)
            .appending(path: "Effects.localized", directoryHint: .isDirectory)
            .appending(path: Self.effectCategory, directoryHint: .isDirectory)
            .appending(path: Self.effectCategory, directoryHint: .isDirectory)
    }

    // MARK: - Freshness check

    /// Compares the bundled and installed templates byte-for-byte. Motion
    /// Template folders are small so a direct comparison is quick and lets
    /// us skip the remove+copy pair when nothing has changed.
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
                    info: "The '\(pathDescription)' folder could not be created. \(error.localizedDescription)"
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
