//
//  AnalysisSnapshotTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Invariants for the value-type snapshot the SwiftUI inspector header
//  displays. SwiftUI is exercised via FxPlug's NSHostingView at runtime and
//  isn't reachable from the Swift Package, so we cover the state surface
//  the header reads from here.
//

import Testing
@testable import CorridorKeyToolboxLogic

@Suite("CorridorKeyAnalysisSnapshot")
struct CorridorKeyAnalysisSnapshotTests {

    @Test("Empty snapshot reports no progress and not working")
    func emptySnapshot() {
        let snapshot = CorridorKeyAnalysisSnapshot.empty
        #expect(snapshot.state == .notAnalysed)
        #expect(snapshot.progress == 0)
        #expect(snapshot.isWorking == false)
    }

    @Test("Progress is analysed / total")
    func progressFraction() {
        let snapshot = CorridorKeyAnalysisSnapshot(
            state: .running,
            analyzedFrameCount: 150,
            totalFrameCount: 300,
            inferenceResolution: 1024
        )
        #expect(abs(snapshot.progress - 0.5) < 1e-9)
    }

    @Test("Progress never exceeds 1 even when analysed overshoots total")
    func progressIsClamped() {
        let snapshot = CorridorKeyAnalysisSnapshot(
            state: .completed,
            analyzedFrameCount: 350,
            totalFrameCount: 300,
            inferenceResolution: 1024
        )
        #expect(snapshot.progress == 1.0)
    }

    @Test("Progress defaults to zero when total is missing")
    func progressFallsBackToZeroWithoutTotal() {
        let snapshot = CorridorKeyAnalysisSnapshot(
            state: .requested,
            analyzedFrameCount: 0,
            totalFrameCount: 0,
            inferenceResolution: 1024
        )
        #expect(snapshot.progress == 0)
    }

    @Test("isWorking covers only requested and running",
          arguments: [
            (CorridorKeyAnalysisSnapshot.State.notAnalysed, false),
            (.requested, true),
            (.running, true),
            (.completed, false),
            (.interrupted, false)
          ])
    func isWorkingMatchesState(state: CorridorKeyAnalysisSnapshot.State, expected: Bool) {
        let snapshot = CorridorKeyAnalysisSnapshot(
            state: state,
            analyzedFrameCount: 0,
            totalFrameCount: 0,
            inferenceResolution: 1024
        )
        #expect(snapshot.isWorking == expected)
    }
}
