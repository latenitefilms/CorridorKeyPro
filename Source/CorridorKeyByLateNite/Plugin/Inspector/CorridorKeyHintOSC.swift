//
//  CorridorKeyHintOSC.swift
//  CorridorKey by LateNite
//
//  Sibling FxPlug plug-in that conforms to `FxOnScreenControl_v4` and
//  draws the hint-point overlay on the Final Cut Pro canvas. Mirrors
//  the Standalone Editor's Subject Hints menu.
//
//  Two-tier interaction model — modelled on Metaburner's hit-test +
//  drag pattern:
//
//    * Subject marker (one yellow-ringed point at the saved Subject
//      Position). Click on it → drag it. The drag updates the
//      `Subject Position` Point parameter so the inspector's X / Y
//      sliders mirror the marker's location and vice-versa. Hidden
//      when "Show Subject Marker" is off.
//
//    * Hint points (zero or more green / red dots at user-placed
//      foreground / background hints). Click in empty canvas space
//      to drop a foreground hint, ⇧-click for background, ⌥-click
//      to erase the nearest hint within tolerance. The hint set
//      flows into the same `Subject Points` custom parameter the
//      Standalone Editor's OSC writes to, so a clip's hints survive
//      a round-trip between the two surfaces.
//
//  Drawing is a render pass into the OSC destination texture FCP
//  supplies (FxPlug's destination is render-target-only so compute
//  writes silently produce nothing). The shared
//  `corridorKeyDrawOSCFragment` shader handles the three marker
//  styles via its `kind` field — 0 = foreground hint (dark disc
//  with a white person silhouette inside, mirroring the editor's
//  `person.fill` icon), 1 = background hint (rounded swatch in the
//  current screen colour, mirroring the editor's screen-colour
//  rectangle), 2 = subject marker (yellow ring + dark fill).
//

import Foundation
import AppKit
import CoreMedia
import Metal

/// Active-part identifiers passed back to FxPlug. Non-zero values
/// indicate a hit; `0` means the click landed in empty canvas
/// space.
private enum HitPart: Int {
    case subjectMarker = 1
    /// `hintBase` is the lowest part-id we ever return for a hint
    /// point — `2` keeps it safely clear of `subjectMarker = 1`.
    /// Per-hint ids are `hintBase + index`.
    static let hintBase: Int = 2
}

@objc(CorridorKeyHintOSC)
class CorridorKeyHintOSC: NSObject, FxOnScreenControl_v4 {

    private let apiManager: any PROAPIAccessing
    /// Tolerance for option-click erase + nearest-hint highlight,
    /// in object-normalised units. Matches the Standalone Editor.
    private static let eraseTolerance: Double = 0.05
    /// Hit radius for the subject marker, in object-normalised
    /// units. Matches the visual ring's outer edge so the marker
    /// is grabbable by clicking anywhere on or just outside it.
    private static let subjectMarkerHitRadius: Double = 0.04

    /// Drag state. `lastObjectPosition` records the user's pointer
    /// in object-normalised space at the most recent move; the
    /// dragged handler computes the delta against it and writes
    /// that into the Subject Position parameter. Mirrors
    /// MetaburnerOSC's lock-protected pattern.
    private let dragLock = NSLock()
    private var lastObjectPosition: CGPoint = CGPoint(x: -1, y: -1)
    private var dragging: Bool = false

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
        // normalised space when reading / writing parameters.
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
        let subjectAnchor = subjectPosition(at: time)
        let screen = currentScreenColor(at: time)

        // Always draw at least the subject marker when the OSC is
        // enabled — without a visible affordance Final Cut Pro
        // stops routing canvas mouse events to the OSC, which is
        // how the marker-drag regression slipped in.
        var points: [PackedPoint] = []
        points.append(
            PackedPoint(
                x: Float(subjectAnchor.x),
                y: Float(subjectAnchor.y),
                radius: Float(Self.subjectMarkerHitRadius * 1.6),
                kind: 2
            )
        )
        for hint in hints.points {
            points.append(
                PackedPoint(
                    x: Float(hint.x),
                    y: Float(hint.y),
                    radius: Float(hint.radiusNormalized),
                    kind: Int32(hint.kind.rawValue)
                )
            )
        }
        do {
            try renderMarkers(
                destinationImage: destinationImage,
                points: points,
                activePart: activePart,
                screenColor: screen
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
        let object = objectPosition(forCanvasX: x, canvasY: y)

        // Subject marker takes priority over hint points so the
        // user can always grab it for a drag, even if a hint sits
        // close to where it's currently parked.
        let subjectAnchor = subjectPosition(at: time)
        let dxSubject = subjectAnchor.x - object.x
        let dySubject = subjectAnchor.y - object.y
        if dxSubject * dxSubject + dySubject * dySubject <= Self.subjectMarkerHitRadius * Self.subjectMarkerHitRadius {
            activePart.pointee = HitPart.subjectMarker.rawValue
            return
        }

        let hints = currentHintSet(at: time)
        var nearest: Int = 0
        var nearestDistanceSquared = Self.eraseTolerance * Self.eraseTolerance
        for (index, point) in hints.points.enumerated() {
            let dx = point.x - object.x
            let dy = point.y - object.y
            let distanceSquared = dx * dx + dy * dy
            if distanceSquared <= nearestDistanceSquared {
                nearestDistanceSquared = distanceSquared
                // First subject marker = part 1, then hints from
                // part 2 onward. The shader's active-part logic
                // matches the index of the point in the array, so
                // hint index 0 occupies array index 1 (after the
                // subject marker) → activePart 2.
                nearest = HitPart.hintBase + index
            }
        }
        activePart.pointee = nearest
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
        if activePart == HitPart.subjectMarker.rawValue {
            // Begin a drag on the subject marker. We don't write
            // anything yet — the actual position update happens in
            // `mouseDragged` so single-click without movement is a
            // no-op. Recording the click position in object space
            // lets the dragged handler compute deltas without
            // re-reading the mouse on every tick.
            let object = objectPosition(forCanvasX: x, canvasY: y)
            beginSubjectMarkerDrag(at: object)
            forceUpdate.pointee = ObjCBool(true)
            return
        }

        // Empty canvas (or over a hint with the option modifier) —
        // place / erase a hint point.
        let object = objectPosition(forCanvasX: x, canvasY: y)
        var hints = currentHintSet(at: time)
        switch hintAction(for: modifiers) {
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
        guard activePart == HitPart.subjectMarker.rawValue, isDraggingSubjectMarker else {
            // Hint points are click-to-place — we don't drag them,
            // so any drag stream that didn't start on the marker is
            // a no-op.
            forceUpdate.pointee = ObjCBool(false)
            return
        }
        let object = objectPosition(forCanvasX: x, canvasY: y)
        let delta = subjectMarkerDragDelta(updatingTo: object)
        guard delta.x != 0 || delta.y != 0 else {
            forceUpdate.pointee = ObjCBool(false)
            return
        }
        let current = subjectPosition(at: time)
        let next = (
            x: min(max(current.x + delta.x, 0), 1),
            y: min(max(current.y + delta.y, 0), 1)
        )
        writeSubjectPosition(x: next.x, y: next.y, at: time)
        forceUpdate.pointee = ObjCBool(true)
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
        endSubjectMarkerDrag()
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

    // MARK: - Subject-marker drag state

    private var isDraggingSubjectMarker: Bool {
        dragLock.lock()
        defer { dragLock.unlock() }
        return dragging
    }

    private func beginSubjectMarkerDrag(at object: (x: Double, y: Double)) {
        dragLock.lock()
        defer { dragLock.unlock() }
        lastObjectPosition = CGPoint(x: object.x, y: object.y)
        dragging = true
    }

    private func subjectMarkerDragDelta(updatingTo object: (x: Double, y: Double)) -> (x: Double, y: Double) {
        dragLock.lock()
        defer { dragLock.unlock() }
        let dx = object.x - Double(lastObjectPosition.x)
        let dy = object.y - Double(lastObjectPosition.y)
        lastObjectPosition = CGPoint(x: object.x, y: object.y)
        return (dx, dy)
    }

    private func endSubjectMarkerDrag() {
        dragLock.lock()
        defer { dragLock.unlock() }
        dragging = false
        lastObjectPosition = CGPoint(x: -1, y: -1)
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

    private func subjectPosition(at time: CMTime) -> (x: Double, y: Double) {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            return (0.5, 0.5)
        }
        var x: Double = 0.5
        var y: Double = 0.5
        retrieval.getXValue(&x, yValue: &y, fromParameter: ParameterIdentifier.subjectPosition, at: time)
        return (x, y)
    }

    /// Reads the active Screen Colour parameter so the background-hint
    /// markers can be tinted with the same green / blue swatch the
    /// Standalone Editor's `HintPointMarker.backgroundMarker` uses —
    /// otherwise the FxPlug's marker visually disagrees with the
    /// Standalone Editor's icon styling and the user has to remember
    /// which colour means "drop this region" on each surface.
    private func currentScreenColor(at time: CMTime) -> ScreenColor {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            return .green
        }
        var raw: Int32 = Int32(ScreenColor.green.rawValue)
        retrieval.getIntValue(&raw, fromParameter: ParameterIdentifier.screenColor, at: time)
        return ScreenColor(rawValue: Int(raw)) ?? .green
    }

    /// Writes the dragged subject-marker position back to the
    /// `Subject Position` parameter. Called from inside `mouseDragged`,
    /// which the host has already wrapped in a parameter-write action
    /// scope — wrapping it again with `startAction` produces the
    /// `[-FxCustomParameterActionAPI startAction:] at an inappropriate
    /// time` warning the renderer log flooded with, and after enough
    /// of those FCP severs the XPC connection. Plain `setXValue`
    /// matches MetaburnerOSC's mouseDragged behaviour.
    private func writeSubjectPosition(x: Double, y: Double, at time: CMTime) {
        guard let setter = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            return
        }
        setter.setXValue(x, yValue: y, toParameter: ParameterIdentifier.subjectPosition, at: time)
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

    /// Writes the updated hint set back to the `Subject Points`
    /// custom parameter. Same caveat as `writeSubjectPosition` —
    /// always invoked from a host mouse callback, so we must not
    /// re-open an action scope.
    private func writeHintSet(_ hints: HintPointSet, at time: CMTime) {
        guard let setter = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            return
        }
        let dict = hints.asParameterDictionary()
        setter.setCustomParameterValue(dict, toParameter: ParameterIdentifier.subjectPoints, at: time)
    }

    // MARK: - Coordinate conversion

    /// Translates canvas pixel coordinates into object-normalised
    /// `(0…1)` so we can write hint / subject parameters.
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

    /// Wire-format point sent to the OSC fragment shader.
    /// `kind = 0` → foreground hint (dark disc with white person
    /// silhouette and white outline ring), `1` → background hint
    /// (rounded screen-colour swatch with black border, matching the
    /// Standalone Editor's icon-style markers), `2` → subject marker
    /// (yellow ring + dark fill).
    private struct PackedPoint {
        var x: Float32
        var y: Float32
        var radius: Float32
        var kind: Int32
    }

    /// Renders every supplied marker via the shared
    /// `corridorKeyDrawOSCFragment` shader. Active-part tracking
    /// matches the array index so hover highlight follows whichever
    /// marker `hitTestOSC` reported. `screenColor` is forwarded to
    /// the shader so the background-hint swatch is drawn in the
    /// user's chosen screen colour.
    private func renderMarkers(
        destinationImage: FxImageTile,
        points: [PackedPoint],
        activePart: Int,
        screenColor: ScreenColor
    ) throws {
        guard !points.isEmpty else { return }

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
        commandBuffer.label = "CorridorKey by LateNite OSC"

        let passDescriptor = MTLRenderPassDescriptor()
        passDescriptor.colorAttachments[0].texture = texture
        passDescriptor.colorAttachments[0].loadAction = .clear
        passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        passDescriptor.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "OSC Markers"

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

        var packed = points
        var pointCount: Int32 = Int32(packed.count)
        var activePart32: Int32 = Int32(activePart)
        var screenColourLinear = Self.linearScreenColour(for: screenColor)
        packed.withUnsafeMutableBytes { rawBytes in
            if let base = rawBytes.baseAddress {
                encoder.setFragmentBytes(base, length: rawBytes.count, index: 0)
            }
        }
        encoder.setFragmentBytes(&pointCount, length: MemoryLayout<Int32>.size, index: 1)
        encoder.setFragmentBytes(&activePart32, length: MemoryLayout<Int32>.size, index: 2)
        // SIMD3<Float> has a 16-byte stride on Metal — the shader
        // reads it as `float3` (also 16-byte aligned), so the layout
        // matches once we hand over the in-memory representation.
        encoder.setFragmentBytes(&screenColourLinear, length: MemoryLayout<SIMD3<Float>>.stride, index: 3)

        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilScheduled()
    }

    /// Background-hint swatch colours. Values mirror the SwiftUI
    /// `screenColorSwatch` palette in `OnScreenControlOverlay` so the
    /// FxPlug OSC and the Standalone Editor mark out a "drop this
    /// region" tile in identical green / blue.
    private static func linearScreenColour(for screenColor: ScreenColor) -> SIMD3<Float> {
        switch screenColor {
        case .green:
            return SIMD3<Float>(0.20, 0.78, 0.30)
        case .blue:
            return SIMD3<Float>(0.20, 0.42, 0.95)
        }
    }
}
