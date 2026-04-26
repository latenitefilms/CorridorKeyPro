//
//  CorridorKeyHintOSC.swift
//  Corridor Key Toolbox
//
//  Sibling FxPlug plug-in that conforms to `FxOnScreenControl_v4` and
//  draws a draggable subject marker on the Final Cut Pro canvas. The
//  marker position is wired to the main filter's `subjectPosition`
//  Point parameter (X/Y in object-normalised 0…1 space), so the user
//  can drag the marker on the canvas OR enter X/Y directly in the
//  inspector — both update the same underlying parameter, mirroring
//  the working pattern from the Metaburner FxPlug.
//
//  Drawing is a render pass (not a compute kernel) into the OSC
//  destination texture FCP supplies; FxPlug's destination is render-
//  target-only so compute writes silently produce nothing.
//

import Foundation
import AppKit
import CoreMedia
import Metal

/// Active part identifiers — non-zero so a hit-test result of `0`
/// unambiguously means "missed every control".
private enum HitPart: Int {
    case marker = 1
}

@objc(CorridorKeyHintOSC)
class CorridorKeyHintOSC: NSObject, FxOnScreenControl_v4 {

    private let apiManager: any PROAPIAccessing
    private let dragLock = NSLock()
    /// True while the user is mid-drag on the marker. Used to ignore
    /// hover-driven `mouseMoved` callbacks that would otherwise fight
    /// the drag.
    private var dragging: Bool = false

    @objc(initWithAPIManager:)
    required init?(apiManager: any PROAPIAccessing) {
        self.apiManager = apiManager
        super.init()
        PluginLog.notice("CorridorKeyHintOSC instantiated by FCP — OSC is registered.")
    }

    // MARK: - FxOnScreenControl_v4

    @objc func drawingCoordinates() -> FxDrawingCoordinates {
        // We work in canvas pixel coordinates because the OSC API
        // returns mouse positions in canvas space and the host's
        // `convertPoint(fromSpace:…)` lets us round-trip into
        // object-normalised space when reading/writing the Point
        // parameter. Matches Metaburner's working setup.
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
        let (objectX, objectY) = subjectPosition(at: time)
        do {
            try renderMarker(
                destinationImage: destinationImage,
                objectX: objectX,
                objectY: objectY,
                isActive: activePart == HitPart.marker.rawValue || isDragging,
                time: time
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
        let canvasMarker = canvasPosition(forObjectPosition: subjectPosition(at: time))
        let dx = canvasMarker.x - x
        let dy = canvasMarker.y - y
        let distance = (dx * dx + dy * dy).squareRoot()
        // Hit radius matches the visual ring's outer edge so the user
        // can grab the marker by clicking anywhere on or just outside
        // the ring.
        let hitRadius = canvasHitRadius()
        activePart.pointee = (distance <= hitRadius) ? HitPart.marker.rawValue : 0
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
        // Whether the click hit the marker or the empty canvas, snap
        // the marker to the click position and start a drag. This is
        // the most discoverable interaction model — users find the
        // marker by clicking anywhere they want it.
        let object = objectPosition(forCanvasX: x, canvasY: y)
        writeSubjectPosition(x: object.x, y: object.y, at: time)
        beginDrag()
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
        guard isDragging else {
            forceUpdate.pointee = ObjCBool(false)
            return
        }
        let object = objectPosition(forCanvasX: x, canvasY: y)
        writeSubjectPosition(x: object.x, y: object.y, at: time)
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
        endDrag()
        forceUpdate.pointee = ObjCBool(true)
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

    // MARK: - Drag state

    private var isDragging: Bool {
        dragLock.lock()
        defer { dragLock.unlock() }
        return dragging
    }

    private func beginDrag() {
        dragLock.lock()
        dragging = true
        dragLock.unlock()
    }

    private func endDrag() {
        dragLock.lock()
        dragging = false
        dragLock.unlock()
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

    private func writeSubjectPosition(x: Double, y: Double, at time: CMTime) {
        let clampedX = max(0, min(1, x))
        let clampedY = max(0, min(1, y))
        guard let actionAPI = apiManager.api(for: (any FxCustomParameterActionAPI_v4).self) as? any FxCustomParameterActionAPI_v4 else {
            return
        }
        actionAPI.startAction(self)
        defer { actionAPI.endAction(self) }
        guard let setter = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            return
        }
        setter.setXValue(clampedX, yValue: clampedY, toParameter: ParameterIdentifier.subjectPosition, at: time)
    }

    // MARK: - Coordinate conversion

    /// Translates an object-normalised `(0…1)` position into canvas
    /// pixel space using FCP's coordinate-conversion API.
    private func canvasPosition(forObjectPosition object: (x: Double, y: Double)) -> (x: Double, y: Double) {
        guard let oscAPI = apiManager.api(for: (any FxOnScreenControlAPI_v4).self) as? any FxOnScreenControlAPI_v4 else {
            return (object.x, object.y)
        }
        var canvasX: Double = 0
        var canvasY: Double = 0
        oscAPI.convertPoint(
            fromSpace: FxDrawingCoordinates(kFxDrawingCoordinates_OBJECT),
            fromX: object.x,
            fromY: object.y,
            toSpace: FxDrawingCoordinates(kFxDrawingCoordinates_CANVAS),
            toX: &canvasX,
            toY: &canvasY
        )
        return (canvasX, canvasY)
    }

    /// Inverse: translates canvas pixel coordinates into object-
    /// normalised `(0…1)` so we can write the Point parameter.
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

    /// Hit radius for the marker in canvas pixels. Scales with the
    /// canvas zoom so the marker stays clickable when the user zooms
    /// out of the viewer.
    private func canvasHitRadius() -> Double {
        guard let oscAPI = apiManager.api(for: (any FxOnScreenControlAPI_v3).self) as? any FxOnScreenControlAPI_v3 else {
            return 32
        }
        let zoom = oscAPI.canvasZoom()
        return max(20, 32 * zoom)
    }

    // MARK: - Drawing

    /// Renders the marker via a render pass. Vertex/fragment shaders
    /// in `CorridorKeyShaders.metal` (`corridorKeyDrawOSCVertex` and
    /// `corridorKeyDrawOSCFragment`) draw a small ring + crosshair
    /// at the supplied object-normalised position.
    private func renderMarker(
        destinationImage: FxImageTile,
        objectX: Double,
        objectY: Double,
        isActive: Bool,
        time: CMTime
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
        commandBuffer.label = "Corridor Key Toolbox OSC Marker"

        let passDescriptor = MTLRenderPassDescriptor()
        passDescriptor.colorAttachments[0].texture = texture
        passDescriptor.colorAttachments[0].loadAction = .clear
        passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        passDescriptor.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "OSC Marker"

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

        // Pack a single point with the subject position. The fragment
        // shader treats `kind = 0` as "foreground" (green dot); we
        // override with white-ring style via a tighter radius so the
        // marker reads as a control point rather than a hint dot.
        // FxPlug's Point parameter convention has y=0 at the BOTTOM,
        // but the OSC fragment shader's UV space has y=0 at the TOP.
        // Flip y here so the marker draws where the user dragged it.
        struct PackedPoint {
            var x: Float32
            var y: Float32
            var radius: Float32
            var kind: Int32
        }
        var packed = [
            PackedPoint(
                x: Float(objectX),
                y: Float(1.0 - objectY),
                radius: 0.05,
                kind: isActive ? 1 : 0
            )
        ]
        var pointCount: Int32 = 1
        var activePart32: Int32 = isActive ? 1 : 0
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
