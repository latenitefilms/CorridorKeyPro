//
//  MetalPreviewView.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  SwiftUI bridge for displaying the keyed preview frame on a
//  `MTKView`. Updates whenever the editor view model publishes a new
//  `PreviewFrame`.
//
//  We use `MTKView` instead of rolling our own `CAMetalLayer`-backed
//  `NSView` because MTKView's display loop is what actually drives the
//  drawable presentation cycle on macOS — overriding `display()` on a
//  raw `NSView` doesn't reliably trigger frame presentation, and the
//  symptoms ("preview is black, parameters are responsive") are very
//  hard to debug. With `MTKView` plus `enableSetNeedsDisplay = true`
//  + `isPaused = true` we get exact, event-driven updates: every time
//  the view model publishes a new texture we call `setNeedsDisplay()`
//  and MTKView calls `draw(in:)` on its delegate.
//
//  Letterboxing is handled in the textured-quad shaders so the
//  inspector pane width can change freely without distorting the
//  preview.
//

import AppKit
import Metal
import MetalKit
import SwiftUI

struct MetalPreviewView: NSViewRepresentable {
    let device: any MTLDevice
    let frame: PreviewFrame?
    let aspectFitSize: CGSize
    /// User-selected backdrop drawn behind the keyed preview image.
    /// Right-click on the preview surfaces a picker that mutates this
    /// value through the editor view model.
    let backdrop: PreviewBackdrop

    func makeCoordinator() -> Coordinator {
        Coordinator(device: device)
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        view.clearColor = MTLClearColorMake(0.05, 0.05, 0.05, 1.0)
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.delegate = context.coordinator
        view.autoResizeDrawable = true
        view.layer?.isOpaque = true
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.update(
            sourceTexture: frame?.texture,
            aspectFitSize: aspectFitSize,
            backdrop: backdrop,
            previewFrame: frame
        )
        nsView.setNeedsDisplay(nsView.bounds)
    }

    /// MTKViewDelegate that owns the textured-quad render pipeline and
    /// the references that must outlive the GPU work driving each
    /// frame (the source `MTLTexture`, its bridging `CVMetalTexture`,
    /// and the originating `PreviewFrame`).
    final class Coordinator: NSObject, MTKViewDelegate {
        let device: any MTLDevice
        let commandQueue: any MTLCommandQueue
        let pipelineState: any MTLRenderPipelineState
        let checkerPipelineState: any MTLRenderPipelineState

        private var sourceTexture: (any MTLTexture)?
        private var aspectFitSize: CGSize = .zero
        private var backdrop: PreviewBackdrop = .checkerboard
        /// Holding the whole `PreviewFrame` keeps the IOSurface backing
        /// the texture alive for the lifetime of the GPU work that
        /// draws it.
        private var heldFrame: PreviewFrame?

        init(device: any MTLDevice) {
            self.device = device
            guard let queue = device.makeCommandQueue() else {
                fatalError("CorridorKey by LateNite preview view could not create a Metal command queue.")
            }
            self.commandQueue = queue
            let library: any MTLLibrary
            do {
                library = try Self.makeShaderLibrary(device: device)
            } catch {
                fatalError("CorridorKey by LateNite preview view could not compile its inline shaders: \(error.localizedDescription)")
            }
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.label = "Corridor Key Standalone Preview"
            pipelineDescriptor.vertexFunction = library.makeFunction(name: "previewVertex")
            pipelineDescriptor.fragmentFunction = library.makeFunction(name: "previewFragment")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            // The compose shader writes the keyed image as
            // **premultiplied alpha** (`float4(foreground * alpha,
            // alpha)` for the Processed output mode). Pair that with
            // standard premultiplied source-over blending so the
            // chosen backdrop shows through where the matte is
            // transparent — without this, a premultiplied-zero
            // background gets drawn as solid black over the chequer
            // pattern / colour fill.
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
            pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            do {
                self.pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            } catch {
                fatalError("CorridorKey by LateNite preview view could not build its render pipeline: \(error.localizedDescription)")
            }
            let checkerDescriptor = MTLRenderPipelineDescriptor()
            checkerDescriptor.label = "Corridor Key Standalone Preview Checker"
            checkerDescriptor.vertexFunction = library.makeFunction(name: "previewVertex")
            checkerDescriptor.fragmentFunction = library.makeFunction(name: "previewCheckerFragment")
            checkerDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            do {
                self.checkerPipelineState = try device.makeRenderPipelineState(descriptor: checkerDescriptor)
            } catch {
                fatalError("CorridorKey by LateNite preview view could not build its checker pipeline: \(error.localizedDescription)")
            }
            super.init()
        }

        func update(
            sourceTexture: (any MTLTexture)?,
            aspectFitSize: CGSize,
            backdrop: PreviewBackdrop,
            previewFrame: PreviewFrame?
        ) {
            self.sourceTexture = sourceTexture
            self.aspectFitSize = aspectFitSize
            self.backdrop = backdrop
            self.heldFrame = previewFrame
        }

        // MARK: - MTKViewDelegate

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // No-op: layout updates the next time `draw(in:)` runs.
        }

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let descriptor = view.currentRenderPassDescriptor,
                  let commandBuffer = commandQueue.makeCommandBuffer()
            else { return }
            commandBuffer.label = "Corridor Key Standalone Preview"

            descriptor.colorAttachments[0].clearColor = clearColorForBackdrop()
            descriptor.colorAttachments[0].loadAction = .clear
            descriptor.colorAttachments[0].storeAction = .store

            encodeQuad(into: commandBuffer, descriptor: descriptor, drawableSize: CGSize(
                width: CGFloat(drawable.texture.width),
                height: CGFloat(drawable.texture.height)
            ))
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }

        /// Test-only entry point: draws one frame into the supplied
        /// `target` texture using the same pipeline / blend state /
        /// backdrop logic the real `draw(in:)` path uses, then
        /// blocks until the GPU finishes so the caller can read the
        /// pixels back. The unit tests use this to assert that the
        /// premultiplied compose output composites correctly over
        /// each backdrop without standing up a real `MTKView`.
        func renderForTesting(into target: any MTLTexture) throws {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw MetalPreviewBackingError.commandBufferUnavailable
            }
            commandBuffer.label = "Corridor Key Standalone Preview (test)"

            let descriptor = MTLRenderPassDescriptor()
            descriptor.colorAttachments[0].texture = target
            descriptor.colorAttachments[0].clearColor = clearColorForBackdrop()
            descriptor.colorAttachments[0].loadAction = .clear
            descriptor.colorAttachments[0].storeAction = .store

            encodeQuad(into: commandBuffer, descriptor: descriptor, drawableSize: CGSize(
                width: CGFloat(target.width),
                height: CGFloat(target.height)
            ))
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw error
            }
        }

        /// Shared encode path. Lays out the quad inside the
        /// drawable's bounds (or the test target's bounds), draws
        /// the chequerboard pattern when that backdrop is selected,
        /// then samples the keyed source onto the same quad with
        /// premultiplied source-over blending. Both `draw(in:)` and
        /// `renderForTesting(into:)` call this so the test suite
        /// covers exactly the same shader path the live preview
        /// uses.
        private func encodeQuad(
            into commandBuffer: any MTLCommandBuffer,
            descriptor: MTLRenderPassDescriptor,
            drawableSize: CGSize
        ) {
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
                return
            }
            encoder.label = "Corridor Key Standalone Preview Quad"

            let quadRect = aspectFittedRect(
                for: aspectFitSize == .zero ? drawableSize : aspectFitSize,
                in: drawableSize
            )

            if backdrop == .checkerboard {
                encoder.setRenderPipelineState(checkerPipelineState)
                var quad = quadVertices(for: quadRect, drawableSize: drawableSize)
                encoder.setVertexBytes(&quad, length: MemoryLayout<PreviewVertex>.stride * 4, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }

            if let texture = sourceTexture {
                encoder.setRenderPipelineState(pipelineState)
                var quad = quadVertices(for: quadRect, drawableSize: drawableSize)
                encoder.setVertexBytes(&quad, length: MemoryLayout<PreviewVertex>.stride * 4, index: 0)
                encoder.setFragmentTexture(texture, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }

            encoder.endEncoding()
        }

        /// Maps the user-selected backdrop to a clear colour that
        /// fills the area outside the aspect-fit quad. The
        /// checkerboard mode picks a neutral dark grey so the
        /// chequer pattern's edges blend into the surrounding
        /// letterbox area; solid-colour modes use the picked
        /// colour directly.
        private func clearColorForBackdrop() -> MTLClearColor {
            switch backdrop {
            case .checkerboard:
                return MTLClearColorMake(0.05, 0.05, 0.05, 1.0)
            case .white:
                return MTLClearColorMake(1.0, 1.0, 1.0, 1.0)
            case .black:
                return MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
            case .yellow:
                return MTLClearColorMake(1.0, 0.85, 0.0, 1.0)
            case .red:
                return MTLClearColorMake(0.95, 0.20, 0.18, 1.0)
            }
        }

        // MARK: - Layout helpers

        private func aspectFittedRect(for content: CGSize, in container: CGSize) -> CGRect {
            guard content.width > 0, content.height > 0,
                  container.width > 0, container.height > 0
            else { return CGRect(origin: .zero, size: container) }
            let scale = min(container.width / content.width, container.height / content.height)
            let fittedSize = CGSize(width: content.width * scale, height: content.height * scale)
            let originX = (container.width - fittedSize.width) * 0.5
            let originY = (container.height - fittedSize.height) * 0.5
            return CGRect(origin: CGPoint(x: originX, y: originY), size: fittedSize)
        }

        private func quadVertices(for rect: CGRect, drawableSize: CGSize) -> [PreviewVertex] {
            // Convert pixel-space rect to clip-space (-1…1).
            let xMin = Float((rect.minX / drawableSize.width) * 2 - 1)
            let xMax = Float((rect.maxX / drawableSize.width) * 2 - 1)
            let yMin = Float((rect.minY / drawableSize.height) * 2 - 1)
            let yMax = Float((rect.maxY / drawableSize.height) * 2 - 1)
            // Texture coordinates: render textures are top-down already
            // (Metal default), so map (0,0) to top-left of the rect.
            return [
                PreviewVertex(position: SIMD2<Float>(xMin, -yMin), uv: SIMD2<Float>(0, 0)),
                PreviewVertex(position: SIMD2<Float>(xMax, -yMin), uv: SIMD2<Float>(1, 0)),
                PreviewVertex(position: SIMD2<Float>(xMin, -yMax), uv: SIMD2<Float>(0, 1)),
                PreviewVertex(position: SIMD2<Float>(xMax, -yMax), uv: SIMD2<Float>(1, 1))
            ]
        }

        // MARK: - Inline shader library

        private static func makeShaderLibrary(device: any MTLDevice) throws -> any MTLLibrary {
            let source = """
            #include <metal_stdlib>
            using namespace metal;

            struct PreviewVertex {
                float2 position;
                float2 uv;
            };

            struct PreviewVertexOut {
                float4 position [[position]];
                float2 uv;
            };

            vertex PreviewVertexOut previewVertex(
                const device PreviewVertex *vertices [[buffer(0)]],
                uint vertexID [[vertex_id]]
            ) {
                PreviewVertex vertex_in = vertices[vertexID];
                PreviewVertexOut out;
                out.position = float4(vertex_in.position, 0.0, 1.0);
                out.uv = vertex_in.uv;
                return out;
            }

            fragment float4 previewFragment(
                PreviewVertexOut in [[stage_in]],
                texture2d<float> source [[texture(0)]]
            ) {
                constexpr sampler textureSampler(filter::linear, address::clamp_to_edge);
                // Pass the premultiplied compose output straight
                // through to the blend stage so the alpha channel
                // actually composes against the chosen backdrop.
                // The previous version forced alpha to 1.0, which
                // baked the keyed image's premultiplied black
                // background onto whichever backdrop the user
                // picked.
                return source.sample(textureSampler, in.uv);
            }

            fragment float4 previewCheckerFragment(
                PreviewVertexOut in [[stage_in]]
            ) {
                float2 grid = floor(in.uv * 24.0);
                float chequer = fmod(grid.x + grid.y, 2.0);
                float3 lightCell = float3(0.30, 0.30, 0.30);
                float3 darkCell  = float3(0.22, 0.22, 0.22);
                float3 color = mix(darkCell, lightCell, chequer);
                return float4(color, 1.0);
            }
            """
            return try device.makeLibrary(source: source, options: nil)
        }
    }
}

/// Vertex layout used by the preview shaders. Mirrors the Metal struct
/// declaration above; kept side by side with the inline source so a
/// rename in either place fails to compile.
private struct PreviewVertex {
    let position: SIMD2<Float>
    let uv: SIMD2<Float>
}

/// Errors the test-only `renderForTesting(into:)` entry point can
/// surface. Production code paths funnel through `draw(in:)` and
/// silently return on these conditions.
enum MetalPreviewBackingError: Error {
    case commandBufferUnavailable
}
