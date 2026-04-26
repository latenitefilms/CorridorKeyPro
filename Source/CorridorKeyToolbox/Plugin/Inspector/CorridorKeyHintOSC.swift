//
//  CorridorKeyHintOSC.swift
//  Corridor Key Toolbox
//
//  Sibling FxPlug plug-in that conforms to `FxOnScreenControl_v4` and lets
//  artists place foreground / background hint dots directly on the Final
//  Cut Pro canvas. The dots travel through a hidden custom parameter on
//  the main effect (`ParameterIdentifier.subjectPoints`); the renderer
//  reads them back during pre-inference and rasterises them into the
//  hint texture so MLX learns "this is the subject" / "this is screen".
//
//  Mouse model:
//
//  * Click on empty canvas → adds a foreground point (green dot).
//  * Option-click on empty canvas → adds a background point (red dot).
//  * Click on an existing dot → starts a drag (mouse-dragged updates
//    the point's coordinates live).
//  * Shift-click on an existing dot → removes it.
//  * Press 'C' or Delete → clears every point on this clip.
//
//  Drawing is delegated to the same `MetalDeviceCache` the renderer
//  uses, so OSC drawing and inference share one pipeline-state pool.
//

import Foundation
import AppKit
import CoreMedia
import Metal

/// Sibling plug-in registered as a `<FxOnScreenControl>` in the
/// XPC service Info.plist. Exposed to FxPlug as an Objective-C class
/// because the protocol uses Objective-C selector dispatch.
@objc(CorridorKeyHintOSC)
final class CorridorKeyHintOSC: NSObject, FxOnScreenControl_v4 {

    private let apiManager: any PROAPIAccessing

    /// Active drag state. Set on mouse-down when the click hits an
    /// existing point; updated on mouse-dragged; consumed and cleared
    /// on mouse-up. The drag index is into the `HintPointSet.points`
    /// array as it stood at mouse-down — so concurrent edits from
    /// elsewhere are ignored for the duration of the drag.
    private struct DragState {
        let pointIndex: Int
    }
    private let dragLock = NSLock()
    private var dragState: DragState?

    @objc(initWithAPIManager:)
    required init?(apiManager: any PROAPIAccessing) {
        self.apiManager = apiManager
        super.init()
    }

    // MARK: - FxOnScreenControl_v4

    @objc func drawingCoordinates() -> FxDrawingCoordinates {
        // Object space is normalised to the input image (0..1 in both
        // axes) which is exactly the space `HintPoint` is stored in,
        // so we never have to convert in/out of canvas pixels.
        // The enum constant is imported as `Int`; the typedef is
        // `NSUInteger`, hence the explicit conversion.
        return FxDrawingCoordinates(kFxDrawingCoordinates_OBJECT)
    }

    @objc(drawOSCWithWidth:height:activePart:destinationImage:atTime:)
    func drawOSC(
        withWidth width: Int,
        height: Int,
        activePart: Int,
        destinationImage: FxImageTile,
        at time: CMTime
    ) {
        let points = currentHintPointSet().points
        guard !points.isEmpty || activePart != 0 else { return }
        do {
            try renderOSC(
                destinationImage: destinationImage,
                points: points,
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
        let points = currentHintPointSet().points
        var bestIndex = 0
        var bestDistance = Double.greatestFiniteMagnitude
        for (index, point) in points.enumerated() {
            let dx = point.x - x
            let dy = point.y - y
            let distance = (dx * dx + dy * dy).squareRoot()
            // Hit radius slightly larger than the visual radius so a
            // user clicking the edge still grabs the dot.
            let hitRadius = point.radiusNormalized * 0.7
            if distance <= hitRadius && distance < bestDistance {
                bestDistance = distance
                bestIndex = index + 1 // 0 reserved for "canvas" / no-hit
            }
        }
        activePart.pointee = bestIndex
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
        // `FxModifierKeys` is `NSUInteger`; the `kFxModifierKey_*`
        // constants come in as plain `Int`. Convert before masking.
        let isShift = (modifiers & FxModifierKeys(kFxModifierKey_SHIFT)) != 0
        let isOption = (modifiers & FxModifierKeys(kFxModifierKey_OPTION)) != 0

        var set = currentHintPointSet()

        if activePart > 0 {
            let pointIndex = activePart - 1
            if isShift {
                // Shift-click on existing dot → remove it.
                if set.points.indices.contains(pointIndex) {
                    set.points.remove(at: pointIndex)
                }
                writeHintPointSet(set)
                forceUpdate.pointee = ObjCBool(true)
                return
            }
            // Plain or option click on an existing dot → start drag.
            dragLock.lock()
            dragState = DragState(pointIndex: pointIndex)
            dragLock.unlock()
            forceUpdate.pointee = ObjCBool(false)
            return
        }

        // Empty canvas → add a new point.
        let kind: HintPointKind = isOption ? .background : .foreground
        let clampedX = clamp01(x)
        let clampedY = clamp01(y)
        set.add(HintPoint(x: clampedX, y: clampedY, kind: kind))
        writeHintPointSet(set)
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
        dragLock.lock()
        let drag = dragState
        dragLock.unlock()
        guard let drag else {
            forceUpdate.pointee = ObjCBool(false)
            return
        }
        var set = currentHintPointSet()
        guard set.points.indices.contains(drag.pointIndex) else {
            forceUpdate.pointee = ObjCBool(false)
            return
        }
        set.points[drag.pointIndex].x = clamp01(x)
        set.points[drag.pointIndex].y = clamp01(y)
        writeHintPointSet(set)
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
        dragLock.lock()
        dragState = nil
        dragLock.unlock()
        // The most recent dragged value was already written by
        // `mouseDragged`; no extra work needed at mouse-up. Force one
        // more update so the canvas redraws without the drag preview.
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
        // 'C' (clear all), Delete (clear all), Backspace (clear all).
        switch asciiKey {
        case 99, 67, 8, 127:
            var set = currentHintPointSet()
            if set.isEmpty {
                didHandle.pointee = ObjCBool(false)
                forceUpdate.pointee = ObjCBool(false)
                return
            }
            set.clear()
            writeHintPointSet(set)
            didHandle.pointee = ObjCBool(true)
            forceUpdate.pointee = ObjCBool(true)
        default:
            didHandle.pointee = ObjCBool(false)
            forceUpdate.pointee = ObjCBool(false)
        }
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

    // MARK: - Custom-parameter I/O

    private func currentHintPointSet() -> HintPointSet {
        guard let retrieval = apiManager.api(for: (any FxParameterRetrievalAPI_v6).self) as? any FxParameterRetrievalAPI_v6 else {
            return HintPointSet()
        }
        var raw: (any NSCopying & NSObjectProtocol & NSSecureCoding)?
        retrieval.getCustomParameterValue(
            &raw,
            fromParameter: ParameterIdentifier.subjectPoints,
            at: CMTime.zero
        )
        return HintPointSet.fromParameterDictionary(raw as? NSDictionary)
    }

    private func writeHintPointSet(_ set: HintPointSet) {
        guard let actionAPI = apiManager.api(for: (any FxCustomParameterActionAPI_v4).self) as? any FxCustomParameterActionAPI_v4 else {
            return
        }
        actionAPI.startAction(self)
        defer { actionAPI.endAction(self) }
        guard let setter = apiManager.api(for: (any FxParameterSettingAPI_v5).self) as? any FxParameterSettingAPI_v5 else {
            return
        }
        setter.setCustomParameterValue(
            set.asParameterDictionary(),
            toParameter: ParameterIdentifier.subjectPoints,
            at: CMTime.zero
        )
    }

    // MARK: - Drawing

    /// Renders the hint dots into `destinationImage` via the cached
    /// `corridorKeyDrawOSCKernel`. The destination texture is owned by
    /// FxPlug and must be written through a render-target-compatible
    /// path; on Apple Silicon a compute kernel that writes to the
    /// IOSurface-backed texture works because every IOSurface texture
    /// FxPlug hands us has `[.shaderRead, .shaderWrite, .renderTarget]`
    /// usage flags.
    private func renderOSC(
        destinationImage: FxImageTile,
        points: [HintPoint],
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
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalDeviceCacheError.commandBufferCreationFailed
        }
        commandBuffer.label = "Corridor Key Toolbox OSC Draw"
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalDeviceCacheError.commandEncoderCreationFailed
        }
        encoder.label = "OSC Draw"
        encoder.setComputePipelineState(entry.computePipelines.drawOSC)
        encoder.setTexture(texture, index: Int(CKTextureIndexOutput.rawValue))

        struct PackedPoint {
            var x: Float32
            var y: Float32
            var radius: Float32
            var kind: Int32
        }
        var packed: [PackedPoint] = points.map {
            PackedPoint(
                x: Float($0.x),
                y: Float($0.y),
                radius: Float($0.radiusNormalized),
                kind: Int32($0.kind.rawValue)
            )
        }
        var pointCount = Int32(packed.count)
        var activePart32 = Int32(activePart)
        if !packed.isEmpty {
            packed.withUnsafeMutableBytes { rawBytes in
                if let base = rawBytes.baseAddress {
                    encoder.setBytes(base, length: rawBytes.count, index: 0)
                }
            }
        } else {
            // Metal requires SOMETHING bound at index 0 even when count
            // is zero, so push a single zero point that the kernel will
            // ignore (loop condition `i < 0` skips immediately).
            var sentinel = PackedPoint(x: 0, y: 0, radius: 0, kind: 0)
            withUnsafeBytes(of: &sentinel) { bytes in
                if let base = bytes.baseAddress {
                    encoder.setBytes(base, length: bytes.count, index: 0)
                }
            }
        }
        encoder.setBytes(&pointCount, length: MemoryLayout<Int32>.size, index: 1)
        encoder.setBytes(&activePart32, length: MemoryLayout<Int32>.size, index: 2)

        let threadgroupWidth = min(entry.computePipelines.drawOSC.threadExecutionWidth, max(texture.width, 1))
        let threadgroupHeight = max(
            1,
            min(
                entry.computePipelines.drawOSC.maxTotalThreadsPerThreadgroup / max(threadgroupWidth, 1),
                max(texture.height, 1)
            )
        )
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        let threadsPerGrid = MTLSize(width: max(texture.width, 1), height: max(texture.height, 1), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private func clamp01(_ value: Double) -> Double {
        return max(0, min(1, value))
    }
}
