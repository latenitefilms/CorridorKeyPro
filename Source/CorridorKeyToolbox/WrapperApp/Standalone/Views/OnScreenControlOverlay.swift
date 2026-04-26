//
//  OnScreenControlOverlay.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Mirrors Final Cut Pro's on-screen control for the Corridor Key
//  effect: a click target that lets the user drop foreground or
//  background hint dots onto the preview, plus a render layer that
//  draws the dots over the keyed image. The hint set is shared with
//  the FxPlug renderer via `PluginStateData.hintPointSet`, so the
//  same dots flow into MLX's 4th input channel during analysis.
//
//  The overlay sits inside the same `ZStack` as the `MetalPreviewView`
//  and matches the renderer's aspect-fit rectangle so click positions
//  resolve correctly regardless of inspector / window resizing.
//

import SwiftUI

struct OnScreenControlOverlay: View {
    @Bindable var viewModel: EditorViewModel
    /// Logical render size of the loaded clip — used so the click
    /// target sits over the same letterboxed quad the preview shader
    /// draws.
    let renderSize: CGSize

    var body: some View {
        GeometryReader { proxy in
            let fittedRect = aspectFittedRect(for: renderSize, in: proxy.size)
            ZStack(alignment: .topLeading) {
                // Click target — only intercepts hits when an OSC
                // tool is active so the rest of the time the user can
                // still drag the parent window from the preview area.
                if viewModel.oscTool != .disabled {
                    Rectangle()
                        .fill(Color.white.opacity(0.0001))
                        .frame(width: fittedRect.width, height: fittedRect.height)
                        .position(x: fittedRect.midX, y: fittedRect.midY)
                        .contentShape(.rect)
                        .onTapGesture { location in
                            let normalisedX = (location.x - fittedRect.minX) / fittedRect.width
                            let normalisedY = (location.y - fittedRect.minY) / fittedRect.height
                            viewModel.handleOSCClick(
                                atNormalizedPoint: CGPoint(x: normalisedX, y: normalisedY)
                            )
                        }
                        .help(toolHint)
                }

                ForEach(Array(viewModel.state.hintPointSet.points.enumerated()), id: \.offset) { item in
                    HintPointMarker(point: item.element)
                        .position(
                            x: fittedRect.minX + CGFloat(item.element.x) * fittedRect.width,
                            y: fittedRect.minY + CGFloat(item.element.y) * fittedRect.height
                        )
                }
            }
            .allowsHitTesting(viewModel.oscTool != .disabled)
        }
    }

    private var toolHint: String {
        switch viewModel.oscTool {
        case .disabled: return ""
        case .foregroundHint: return "Click to drop a foreground hint."
        case .backgroundHint: return "Click to drop a background hint."
        case .eraseHint: return "Click near a hint to remove it."
        }
    }

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
}

/// Visual marker for a single hint point. Foreground points are green
/// (matches the FCP OSC); background points are red. A subtle dark
/// outline keeps both readable on bright matte areas.
private struct HintPointMarker: View {
    let point: HintPoint

    var body: some View {
        ZStack {
            Circle()
                .strokeBorder(Color.black.opacity(0.5), lineWidth: 2)
                .frame(width: 16, height: 16)
            Circle()
                .fill(fillColor)
                .frame(width: 12, height: 12)
        }
        .shadow(color: .black.opacity(0.4), radius: 1.5, y: 1)
    }

    private var fillColor: Color {
        switch point.kind {
        case .foreground: return Color(red: 0.30, green: 0.85, blue: 0.30)
        case .background: return Color(red: 0.95, green: 0.35, blue: 0.30)
        }
    }
}

/// Compact toolbar that sits alongside the preview and lets the user
/// pick a hint tool. Embedded inside `EditorView` so it travels with
/// the preview area.
struct OSCToolbar: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        HStack(spacing: 8) {
            ForEach(OnScreenControlTool.allCases) { tool in
                Button {
                    viewModel.oscTool = (viewModel.oscTool == tool) ? .disabled : tool
                } label: {
                    Label(tool.displayName, systemImage: tool.systemImage)
                        .labelStyle(.iconOnly)
                        .frame(width: 32, height: 28)
                }
                .buttonStyle(.bordered)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(viewModel.oscTool == tool ? Color.accentColor.opacity(0.25) : Color.clear)
                )
                .help(toolHelpText(for: tool))
            }

            if !viewModel.state.hintPointSet.isEmpty {
                Divider().frame(height: 18)
                Button("Clear", systemImage: "trash") {
                    viewModel.clearAllHints()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Remove every hint point.")
            }
        }
        .padding(8)
        .background(.regularMaterial, in: .rect(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.white.opacity(0.05), lineWidth: 0.5)
        )
        .shadow(color: .black.opacity(0.3), radius: 6, y: 2)
    }

    private func toolHelpText(for tool: OnScreenControlTool) -> String {
        switch tool {
        case .disabled: return "Disable hint placement (default)."
        case .foregroundHint: return "Mark a region as foreground."
        case .backgroundHint: return "Mark a region as background."
        case .eraseHint: return "Erase the closest hint."
        }
    }
}
