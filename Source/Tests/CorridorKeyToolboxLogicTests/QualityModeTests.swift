//
//  QualityModeTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Guards the automatic-quality rung ladder against accidental drift. The
//  plug-in's "Recommended" option is chosen to line up with CorridorKey's OFX
//  reference so editors see identical defaults across hosts — breaking the
//  mapping silently would quietly degrade output quality on real footage.
//

import Testing
@testable import CorridorKeyToolboxLogic

@Suite("QualityMode")
struct QualityModeTests {

    @Test("Automatic mode scales with long edge", arguments: [
        (100, 512),
        (512, 512),
        (1000, 512),
        (1001, 1024),
        (1920, 1024),
        (2000, 1024),
        (2001, 1536),
        (3000, 1536),
        (3001, 2048),
        (4096, 2048),
        (8192, 2048)
    ])
    func automaticResolutionLadder(longEdge: Int, expected: Int) {
        #expect(QualityMode.automatic.resolvedInferenceResolution(forLongEdge: longEdge) == expected)
    }

    @Test("Manual rungs ignore input size")
    func manualRungsAreFixed() {
        let longEdges = [100, 1920, 4096]
        for longEdge in longEdges {
            #expect(QualityMode.draft512.resolvedInferenceResolution(forLongEdge: longEdge) == 512)
            #expect(QualityMode.high1024.resolvedInferenceResolution(forLongEdge: longEdge) == 1024)
            #expect(QualityMode.ultra1536.resolvedInferenceResolution(forLongEdge: longEdge) == 1536)
            #expect(QualityMode.maximum2048.resolvedInferenceResolution(forLongEdge: longEdge) == 2048)
        }
    }

    @Test("Display names are stable for saved documents")
    func displayNamesAreStable() {
        #expect(QualityMode.automatic.displayName == "Recommended")
        #expect(QualityMode.draft512.displayName == "Draft (512)")
        #expect(QualityMode.high1024.displayName == "High (1024)")
        #expect(QualityMode.ultra1536.displayName == "Ultra (1536)")
        #expect(QualityMode.maximum2048.displayName == "Maximum (2048)")
    }
}
