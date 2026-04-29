//
//  CorridorKeyHintOSC.swift
//  CorridorKey by LateNite
//
//  Sibling FxPlug plug-in that conforms to `FxOnScreenControl_v4` and
//  draws the hint-point overlay on the Final Cut Pro canvas. Mirrors
//  the Standalone Editor's Subject Hints menu: foreground points
//  steer the matte toward 1.0 in their neighbourhood, background
//  points toward 0.0, and the user can erase nearest with a
//  modifier-key click. Both interfaces write into the same
//  `HintPointSet` custom parameter so a clip's hints travel with
//  the project.
//
//  Modifier-key click semantics — chosen to match Photoshop /
//  Affinity / DaVinci's brush conventions:
//
//    Plain click          → drop a foreground hint
//    Shift + click        → drop a background hint
//    Option (alt) + click → erase the nearest hint within tolerance
//
//  Drawing is a render pass into the OSC destination texture FCP
//  supplies; FxPlug's destination is render-target-only so compute
//  writes silently produce nothing. The same `corridorKeyDrawOSCVertex`
//  / `Fragment` shader pair the editor previews with renders here so
//  the two surfaces look identical.
//

import Foundation
import AppKit
import CoreMedia
import Metal

@objc(CorridorKeyHintOSC)
class CorridorKeyHintOSC: NSObject, FxOnScreenControl_v4 {

    private let apiManager: any PROAPIAccessing
    /// Tolerance for option-click erase, in object-normalised units.
    /// Matches the Standalone Editor's tolerance so a user with
    /// muscle memory from one host hits the right hint in the other.
    private static let eraseTolerance: Double = 0.05

    @objc(initWithAPIManager:)
    required init?(apiManager: any PROAPIAccessing) {
        self.apiManager = apiManager
        super.init()
        PluginLog.notice("CorridorKeyHintOSC instantiated by FCP — OSC is registered.")
    }

    // MARK: - FxOnScreenControl_v4

    @objc func drawingCoordinates() -> FxDrawingCoordinates {
        // Canvas pixel coordinates because the OSC API returns mouse
        // positions in canvas space; we round-trip into object-
        // normalised space when reading / writing the hint point set.
        return FxDrawingCoordinates(kFxDrawingCoordinates_CANVAS)
    }

    @objc(drawOSCWithWidth:height:activePart:destinationImage:atTime:)
    func drawOSC(
        withWidth width: Int,
        height: Int,
        activePart: Int,
        destinationImage: FxImageTile,
        at time: CMTime
    ) {
        guard subjectMarkerVisible(at: time) else { return }
        let hints = currentHintSet(at: time)
        guard !hints.points.isEmpty else { return }
        do {
            try renderMarkers(
                destinationImage: destinationImage,
                hints: hints,
                activePart: activePart
            )
        } catch {
            PluginLog.error("OSC draw failed: \(error.localizedDescription)")
        }
    }

    @objc(hitTestOSCAtMousePositionX:mousePositionY:activePart:atTime:)
    func hitTestOSC(
        atMousePositionX x: Double,
        mousePositionY y: Double,
        activePart: UnsafeMutablePointer<Int>,
        at time: CMTime
    ) {
        guard subjectMarkerVisible(at: time) else {
            activePart.pointee = 0
            return
        }
        let hits = currentHintSet(at: time)
        guard !hits.points.isEmpty else {
            activePart.pointee = 0
            return
        }
        let object = objectPosition(forCanvasX: x, canvasY: y)
        var nearest: Int = 0
        var nearestDistanceSquared = Double.infinity
        for (index, point) in hits.points.enumerated() {
            let dx = point.x - object.x
            let dy = point.y - object.y
            let distanceSquared = dx * dx + dy * dy
            if distanceSquared < nearestDistanceSquared {
                nearestDistanceSquared = distanceSquared
                nearest = index + 1 // FxPlug expects non-zero for "hit"
            }
        }
        // Use the same tolerance the erase action uses so the active
        // highlight only appears when the user is close enough that
        // option-click would actually delete the hint.
        if nearestDistanceSquared <= Self.eraseTolerance * Self.eraseTolerance {
            activePart.pointee = nearest
        } else {
            activePart.pointee = 0
        }
    }

    @objc(mouseDownAtPositionX:positionY:activePart:modifiers:forceUpdate:atTime:)
    func mouseDown(
        atPositionX x: Double,
        positionY y: Double,
        activePart: Int,
        modifiers: FxModifierKeys,
        forceUpdate: UnsafeMutablePointer<ObjCBool>,
        at time: CMTime
    ) {
        guard subjectMarkerVisible(at: time) else {
            forceUpdate.pointee = ObjCBool(false)
            return
        }
        let object = objectPosition(forCanvasX: x, canvasY: y)
        var hints = currentHintSet(at: time)
        let action = hintAction(for: modifiers)
        switch action {
        case .addForeground:
            hints.add(HintPoint(x: object.x, y: object.y, kind: .foreground))
        case .addBackground:
            hints.add(HintPoint(x: object.x, y: object.y, kind: .background))
        case .eraseNearest:
            let removed = hints.removeNearest(toX: object.x, y: object.y, tolerance: Self.eraseTolerance)
            if !removed {
                forceUpdate.pointee = ObjCBool(false)
                return
            }
        }
        writeHintSet(hints, at: time)
        forceUpdate.pointee = ObjCBool(true)
    }

    @objc(mouseDraggedAtPositionX:positionY:activePart:modifiers:forceUpdate:atTime:)
    func mouseDragged(
        atPositionX x: Double,
        positionY y: Double,
        activePart: Int,
        modifiers: FxModifierKeys,
        forceUpdate: UnsafeMutablePointer<ObjCBool>,
        at time: CMTime
    ) {
        // Hint points are click-to-place (Photoshop brush style); we
        // don't drag them. Reporting `false` here keeps FCP from
        // dispatching a continuous drag stream we'd ignore anyway.
        forceUpdate.pointee = ObjCBool(false)
    }

    @objc(mouseUpAtPositionX:positionY:activePart:modifiers:forceUpdate:atTime:)
    func mouseUp(
        atPositionX x: Double,
        positionY y: Double,
        activePart: Int,
        modifiers: FxModifierKeys,
        forceUpdate: UnsafeMutablePointer<ObjCBool>,
        at time: CMTime
    ) {
        forceUpdate.pointee = ObjCBool(false)
    }

    @objc(keyDownAtPositionX:positionY:keyPressed:modifiers:forceUpdate:didHandle:atTime:)
    func keyDown(
        atPositionX x: Double,
        positionY y: Double,
        keyPressed asciiKey: UInt16,
        modifiers: FxModifierKeys,
        forceUpdate: UnsafeMutablePointer<ObjCBool>,
        didHandle: UnsafeMutablePointer<ObjCBool>,
        at time: CMTime
    ) {
        didHandle.pointee = ObjCBool(false)
        forceUpdate.pointee = ObjCBool(false)
    }

    @objc(keyUpAtPositionX:positionY:keyPressed:modifiers:forceUpdate:didHandle:atTime:)
    func keyUp(
        atPositionX x: Double,
        positionY y: Double,
        keyPressed asciiKey: UInt16,
        modifiers: FxModifierKeys,
        forceUpdate: UnsafeMutablePointer<ObjCBool>,
        didHandle: UnsafeMutablePointer<ObjCBool>,
        at time: CMTime
    ) {
        didHandle.pointee = ObjCBool(false)
        forceUpdate.pointee = ObjCBool(false)
    }

    // MARK: - Tool action

    private enum HintAction {
        case addForeground
        case addBackground
        case eraseNearest
    }

    private func hintAction(for modifiers: FxModifierKeys) -> HintAction {
        // `FxModifierKeys` is a `UInt` bitmask whose bits are defined
        // in `FxOnScreenControl.h`. Option (alt) takes precedence
        // over Shift so option+shift+click still erases — the
        // dominant intent on a multi-modifier press is "remove this
        // dot" rather than "swap kind".
        let raw = UInt(modifiers)
        if raw & UInt(kFxModifierKey_OPTION) != 0 {
            return .eraseNearest
        }
        if raw & UInt(kFxModifierKey_SHIFT) != 0 {
            return .addBackground
        }
        return .addForeground
    }

    // MARK: - Parameter I/O

    private func subjectMarkerVisible(at time: CMTime) -> Bool {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            return true
        }
        var raw = ObjCBool(true)
        retrieval.getBoolValue(&raw, fromParameter: ParameterIdentifier.showSubjectMarker, at: time)
        return raw.boolValue
    }

    private func currentHintSet(at time: CMTime) -> HintPointSet {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            return HintPointSet()
        }
        var raw: (any NSCopying & NSObjectProtocol & NSSecureCoding)?
        retrieval.getCustomParameterValue(
            &raw,
            fromParameter: ParameterIdentifier.subjectPoints,
            at: time
        )
        return HintPointSet.fromParameterDictionary(raw as? NSDictionary)
    }

    private func writeHintSet(_ hints: HintPointSet, at time: CMTime) {
        guard let actionAPI = apiManager.api(for: (any FxCustomParameterActionAPI_v4).self) as? any FxCustomParameterActionAPI_v4 else {
            return
        }
        actionAPI.startAction(self)
        defer { actionAPI.endAction(self) }
        guard let setter = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            return
        }
        let dict = hints.asParameterDictionary()
        setter.setCustomParameterValue(dict, toParameter: ParameterIdentifier.subjectPoints, at: time)
    }

    // MARK: - Coordinate conversion

    /// Translates canvas pixel coordinates into object-normalised
    /// `(0…1)` so we can write the hint set parameter.
    private func objectPosition(forCanvasX x: Double, canvasY y: Double) -> (x: Double, y: Double) {
        guard let oscAPI = apiManager.api(for: (any FxOnScreenControlAPI_v4).self) as? any FxOnScreenControlAPI_v4 else {
            return (x, y)
        }
        var objectX: Double = 0
        var objectY: Double = 0
        oscAPI.convertPoint(
            fromSpace: FxDrawingCoordinates(kFxDrawingCoordinates_CANVAS),
            fromX: x,
            fromY: y,
            toSpace: FxDrawingCoordinates(kFxDrawingCoordinates_OBJECT),
            toX: &objectX,
            toY: &objectY
        )
        return (objectX, objectY)
    }

    // MARK: - Drawing

    /// Renders every hint point as a coloured ring + dot via the
    /// shared `corridorKeyDrawOSCFragment` shader. Foreground points
    /// (`kind = 0`) draw green; background points (`kind = 1`) draw
    /// red. The active-part index passed to the shader brightens the
    /// matching ring so the user can see which hint they're about to
    /// erase / interact with.
    private func renderMarkers(
        destinationImage: FxImageTile,
        hints: HintPointSet,
        activePart: Int
    ) throws {
        let deviceCache = MetalDeviceCache.shared
        guard let device = deviceCache.device(forRegistryID: destinationImage.deviceRegistryID) else {
            throw MetalDeviceCacheError.unknownDevice(destinationImage.deviceRegistryID)
        }
        let entry = try deviceCache.entry(for: device)
        guard let commandQueue = entry.borrowCommandQueue() else {
            throw MetalDeviceCacheError.queueExhausted
        }
        defer { entry.returnCommandQueue(commandQueue) }
        guard let texture = destinationImage.metalTexture(for: device) else {
            throw MetalDeviceCacheError.unknownDevice(destinationImage.deviceRegistryID)
        }
        let renderPipelines = try entry.renderPipelines(for: texture.pixelFormat)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        commandBuffer.label = "CorridorKey by LateNite OSC Hints"

        let passDescriptor = MTLRenderPassDescriptor()
        passDescriptor.colorAttachments[0].texture = texture
        passDescriptor.colorAttachments[0].loadAction = .clear
        passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        passDescriptor.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "OSC Hint Markers"

        let tileWidth = Float(destinationImage.tilePixelBounds.right - destinationImage.tilePixelBounds.left)
        let tileHeight = Float(destinationImage.tilePixelBounds.top - destinationImage.tilePixelBounds.bottom)
        let halfW = tileWidth * 0.5
        let halfH = tileHeight * 0.5
        var vertices: [CKVertex2D] = [
            CKVertex2D(position: SIMD2<Float>(halfW, -halfH), textureCoordinate: SIMD2<Float>(1, 1)),
            CKVertex2D(position: SIMD2<Float>(-halfW, -halfH), textureCoordinate: SIMD2<Float>(0, 1)),
            CKVertex2D(position: SIMD2<Float>(halfW, halfH), textureCoordinate: SIMD2<Float>(1, 0)),
            CKVertex2D(position: SIMD2<Float>(-halfW, halfH), textureCoordinate: SIMD2<Float>(0, 0))
        ]
        var viewportSize = SIMD2<UInt32>(UInt32(tileWidth), UInt32(tileHeight))

        encoder.setViewport(MTLViewport(
            originX: 0, originY: 0,
            width: Double(tileWidth), height: Double(tileHeight),
            znear: -1, zfar: 1
        ))
        encoder.setRenderPipelineState(renderPipelines.drawOSC)
        encoder.setVertexBytes(
            &vertices,
            length: MemoryLayout<CKVertex2D>.stride * vertices.count,
            index: Int(CKVertexInputIndexVertices.rawValue)
        )
        encoder.setVertexBytes(
            &viewportSize,
            length: MemoryLayout<SIMD2<UInt32>>.size,
            index: Int(CKVertexInputIndexViewportSize.rawValue)
        )

        struct PackedPoint {
            var x: Float32
            var y: Float32
            var radius: Float32
            var kind: Int32
        }
        // Pack one entry per stored hint. The shader's `kind` value
        // matches `HintPointKind.rawValue` (0 = foreground / green,
        // 1 = background / red) so no extra translation is needed.
        var packed: [PackedPoint] = hints.points.map { point in
            PackedPoint(
                x: Float(point.x),
                y: Float(point.y),
                radius: 0.04,
                kind: Int32(point.kind.rawValue)
            )
        }
        var pointCount: Int32 = Int32(packed.count)
        var activePart32: Int32 = Int32(activePart)
        packed.withUnsafeMutableBytes { rawBytes in
            if let base = rawBytes.baseAddress {
                encoder.setFragmentBytes(base, length: rawBytes.count, index: 0)
            }
        }
        encoder.setFragmentBytes(&pointCount, length: MemoryLayout<Int32>.size, index: 1)
        encoder.setFragmentBytes(&activePart32, length: MemoryLayout<Int32>.size, index: 2)

        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilScheduled()
    }
}
