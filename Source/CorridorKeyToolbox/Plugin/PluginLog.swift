//
//  PluginLog.swift
//  CorridorKey by LateNite
//
//  File-based logging utility for the FxPlug XPC service. Because Final Cut Pro
//  runs plug-ins in an out-of-process sandbox, messages written with `print` or
//  `NSLog` rarely reach Xcode's console. Routing log output through a durable
//  log file (plus `NSLog` for Console.app) gives us a reliable record for
//  diagnosing issues that only reproduce inside the host.
//

import Foundation

/// Namespace for CorridorKey by LateNite logging. Exposes level-tagged helpers that
/// forward to `NSLog` and append to a persistent log file under
/// `~/Library/Application Support/CorridorKey by LateNite/Logs/`. Calls are
/// intentionally forgiving — the log path exists from the first redirect
/// install, and emits made before that still show up in Console.app.
enum PluginLog {

    /// Installs a `freopen`-style redirect so every write to `stderr` is also
    /// captured to a persistent log file. Safe to call multiple times — any
    /// call after the first is a no-op.
    static func installFileRedirect() {
        guard !didInstallRedirect else { return }
        didInstallRedirect = true

        let fileManager = FileManager.default
        guard let applicationSupport = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) else {
            error("Unable to locate Application Support directory for log file.")
            return
        }

        let logDirectory = applicationSupport
            .appending(path: "CorridorKey by LateNite", directoryHint: .isDirectory)
            .appending(path: "Logs", directoryHint: .isDirectory)
        if !fileManager.fileExists(atPath: logDirectory.path) {
            do {
                try fileManager.createDirectory(
                    at: logDirectory,
                    withIntermediateDirectories: true
                )
            } catch {
                self.error("Could not create log directory at \(logDirectory.path): \(error.localizedDescription)")
                return
            }
        }

        let versionString = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
        let buildString = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0"
        let logFile = logDirectory.appending(path: "CorridorKeyToolboxRenderer-\(versionString)-\(buildString).log")

        freopen(logFile.path.cString(using: .utf8), "a+", stderr)
        notice("--------- CorridorKey by LateNite Renderer \(versionString) (\(buildString)) ---------")
        notice("Log file: \(logFile.path)")
    }

    static func notice(_ message: String) {
        NSLog("[CorridorKey by LateNite] %@", message)
    }

    static func debug(_ message: String) {
        NSLog("[CorridorKey by LateNite] DEBUG: %@", message)
    }

    static func warning(_ message: String) {
        NSLog("[CorridorKey by LateNite] WARNING: %@", message)
    }

    static func error(_ message: String) {
        NSLog("[CorridorKey by LateNite] ERROR: %@", message)
    }

    private nonisolated(unsafe) static var didInstallRedirect = false
}
