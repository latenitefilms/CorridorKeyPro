//
//  InspectorView.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Mirrors the Final Cut Pro inspector layout for the standalone
//  editor: Settings → Interior Detail → Matte → Edge & Spill → Edge
//  Refinement → Temporal Stability. Each group is a sibling `View`
//  struct so SwiftUI invalidates only the group whose data changed.
//
//  Sliders bind to the editor's `PluginStateData` directly via
//  `@Bindable`; flicking a value triggers `parameterDidChange()` on
//  the view model so the preview re-renders.
//

import SwiftUI

struct InspectorView: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 0) {
                InspectorGroupSeparator(isFirst: true)
                AnalysisStatusGroup(viewModel: viewModel)
                InspectorGroupSeparator()
                SettingsGroup(viewModel: viewModel)
                InspectorGroupSeparator()
                InteriorDetailGroup(viewModel: viewModel)
                InspectorGroupSeparator()
                MatteGroup(viewModel: viewModel)
                InspectorGroupSeparator()
                EdgeAndSpillGroup(viewModel: viewModel)
                InspectorGroupSeparator()
                EdgeRefinementGroup(viewModel: viewModel)
                InspectorGroupSeparator()
                TemporalStabilityGroup(viewModel: viewModel)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .background(.regularMaterial)
        .scrollIndicators(.hidden)
    }
}

/// Visual separator between inspector groups. Inset slightly so it
/// reads as a section divider rather than a full-width edge.
private struct InspectorGroupSeparator: View {
    var isFirst: Bool = false

    var body: some View {
        Divider()
            .padding(.top, isFirst ? 8 : 18)
            .padding(.bottom, isFirst ? 14 : 18)
    }
}

// MARK: - Analyse / status group

private struct AnalysisStatusGroup: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        InspectorGroup(title: "Analysis") {
            HStack(spacing: 8) {
                Button("Analyse Clip", systemImage: "wand.and.stars", action: {
                    viewModel.runAnalysis()
                })
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.phase.isReady || viewModel.analysisStatus.inProgress)

                if viewModel.analysisStatus.inProgress {
                    Button("Stop", systemImage: "stop.fill") {
                        viewModel.cancelAnalysis()
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()
            }
            AnalysisProgressLabel(status: viewModel.analysisStatus, totalFrames: viewModel.totalFrames)
            HStack(spacing: 6) {
                Image(systemName: warmupIconName)
                    .foregroundStyle(warmupColor)
                Text(warmupLabel)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                Text("Backend: \(viewModel.renderBackendDescription)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                Image(systemName: "square.stack.3d.up")
                    .foregroundStyle(.secondary)
                Text("Cached mattes: \(viewModel.matteCache.count) / \(viewModel.totalFrames)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var warmupIconName: String {
        switch viewModel.warmupStatus {
        case .cold: return "moon.zzz"
        case .warming: return "clock"
        case .ready: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    private var warmupColor: Color {
        switch viewModel.warmupStatus {
        case .cold: return .secondary
        case .warming: return .orange
        case .ready: return .green
        case .failed: return .red
        }
    }

    private var warmupLabel: String {
        switch viewModel.warmupStatus {
        case .cold: return "Neural model: cold"
        case .warming(let resolution): return "Neural model: loading (\(resolution)px)…"
        case .ready(let resolution): return "Neural model: ready (\(resolution)px)"
        case .failed(let message): return "Neural model failed: \(message)"
        }
    }
}

private struct AnalysisProgressLabel: View {
    let status: EditorAnalysisStatus
    let totalFrames: Int

    var body: some View {
        switch status {
        case .idle:
            Text("Press Analyse Clip to build the matte cache.")
                .font(.callout)
                .foregroundStyle(.secondary)
        case .running(let processed, let total):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: Double(processed), total: Double(max(total, 1)))
                Text("Analysing frame \(processed) of \(total)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .completed(let elapsed, let frames):
            Text("Analysed \(frames) frames in \(elapsed, format: .number.precision(.fractionLength(1))) s")
                .font(.callout)
                .foregroundStyle(.green)
        case .failed(let message):
            Text("Analysis failed: \(message)")
                .font(.callout)
                .foregroundStyle(.red)
        case .cancelled:
            Text("Analysis cancelled.")
                .font(.callout)
                .foregroundStyle(.orange)
        }
    }
}

// MARK: - Setting group

private struct SettingsGroup: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        InspectorGroup(title: "Settings") {
            EnumPicker(
                title: ParameterRanges.qualityModeName,
                selection: $viewModel.state.qualityMode,
                onChange: viewModel.parameterDidChange
            )
            EnumPicker(
                title: ParameterRanges.screenColorName,
                selection: $viewModel.state.screenColor,
                onChange: viewModel.parameterDidChange
            )
            EnumPicker(
                title: ParameterRanges.upscaleMethodName,
                selection: $viewModel.state.upscaleMethod,
                onChange: viewModel.parameterDidChange
            )
            EnumPicker(
                title: ParameterRanges.outputModeName,
                selection: $viewModel.state.outputMode,
                onChange: viewModel.parameterDidChange
            )
            ParameterToggle(
                title: ParameterRanges.autoSubjectHintName,
                isOn: $viewModel.state.autoSubjectHintEnabled,
                onChange: viewModel.parameterDidChange
            )
        }
    }
}

private struct InteriorDetailGroup: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        InspectorGroup(title: "Interior Detail") {
            ParameterToggle(
                title: ParameterRanges.sourcePassthroughName,
                isOn: $viewModel.state.sourcePassthroughEnabled,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.passthroughErode,
                value: $viewModel.state.passthroughErodeNormalized,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.passthroughBlur,
                value: $viewModel.state.passthroughBlurNormalized,
                onChange: viewModel.parameterDidChange
            )
        }
    }
}

private struct MatteGroup: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        InspectorGroup(title: "Matte") {
            ParameterFloatSlider(
                range: ParameterRanges.alphaBlackPoint,
                value: $viewModel.state.alphaBlackPoint,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.alphaWhitePoint,
                value: $viewModel.state.alphaWhitePoint,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.alphaErode,
                value: $viewModel.state.alphaErodeNormalized,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.alphaSoftness,
                value: $viewModel.state.alphaSoftnessNormalized,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.alphaGamma,
                value: $viewModel.state.alphaGamma,
                onChange: viewModel.parameterDidChange
            )
            ParameterToggle(
                title: ParameterRanges.autoDespeckleName,
                isOn: $viewModel.state.autoDespeckleEnabled,
                onChange: viewModel.parameterDidChange
            )
            ParameterIntSlider(
                range: ParameterRanges.despeckleSize,
                value: $viewModel.state.despeckleSize,
                onChange: viewModel.parameterDidChange
            )
            .disabled(!viewModel.state.autoDespeckleEnabled)
            ParameterFloatSlider(
                range: ParameterRanges.refinerStrength,
                value: $viewModel.state.refinerStrength,
                onChange: viewModel.parameterDidChange
            )
        }
    }
}

private struct EdgeAndSpillGroup: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        InspectorGroup(title: "Edge & Spill") {
            ParameterFloatSlider(
                range: ParameterRanges.despillStrength,
                value: $viewModel.state.despillStrength,
                onChange: viewModel.parameterDidChange
            )
            EnumPicker(
                title: ParameterRanges.spillMethodName,
                selection: $viewModel.state.spillMethod,
                onChange: viewModel.parameterDidChange
            )
        }
    }
}

private struct EdgeRefinementGroup: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        InspectorGroup(title: "Edge Refinement") {
            ParameterToggle(
                title: ParameterRanges.lightWrapName,
                isOn: $viewModel.state.lightWrapEnabled,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.lightWrapStrength,
                value: $viewModel.state.lightWrapStrength,
                onChange: viewModel.parameterDidChange
            )
            .disabled(!viewModel.state.lightWrapEnabled)
            ParameterFloatSlider(
                range: ParameterRanges.lightWrapRadius,
                value: $viewModel.state.lightWrapRadius,
                onChange: viewModel.parameterDidChange
            )
            .disabled(!viewModel.state.lightWrapEnabled)

            ParameterToggle(
                title: ParameterRanges.edgeDecontaminateName,
                isOn: $viewModel.state.edgeDecontaminateEnabled,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.edgeDecontaminateStrength,
                value: $viewModel.state.edgeDecontaminateStrength,
                onChange: viewModel.parameterDidChange
            )
            .disabled(!viewModel.state.edgeDecontaminateEnabled)
        }
    }
}

private struct TemporalStabilityGroup: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        InspectorGroup(title: "Temporal Stability") {
            ParameterToggle(
                title: ParameterRanges.temporalStabilityName,
                isOn: $viewModel.state.temporalStabilityEnabled,
                onChange: viewModel.parameterDidChange
            )
            ParameterFloatSlider(
                range: ParameterRanges.temporalStabilityStrength,
                value: $viewModel.state.temporalStabilityStrength,
                onChange: viewModel.parameterDidChange
            )
            .disabled(!viewModel.state.temporalStabilityEnabled)
        }
    }
}

// MARK: - Reusable controls

/// Visual wrapper around an inspector group: the Final Cut Pro
/// inspector uses subtle dividers and bold section headers; we mirror
/// that here with a `DisclosureGroup`-style affordance.
struct InspectorGroup<Content: View>: View {
    let title: String
    let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.primary)
            content
        }
        .padding(.bottom, 4)
    }
}

struct ParameterFloatSlider: View {
    let range: FloatParameterRange
    @Binding var value: Double
    var onChange: () -> Void = {}

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(range.name)
                    .font(.callout)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(2)))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: $value,
                in: range.sliderMin...range.sliderMax,
                step: range.step
            )
            .onChange(of: value) {
                onChange()
            }
        }
    }
}

struct ParameterIntSlider: View {
    let range: IntParameterRange
    @Binding var value: Int
    var onChange: () -> Void = {}

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(range.name)
                    .font(.callout)
                Spacer()
                Text(value, format: .number)
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { Double(value) },
                    set: { value = Int($0.rounded()) }
                ),
                in: Double(range.sliderMin)...Double(range.sliderMax),
                step: Double(range.step)
            )
            .onChange(of: value) {
                onChange()
            }
        }
    }
}

struct ParameterToggle: View {
    let title: String
    @Binding var isOn: Bool
    var onChange: () -> Void = {}

    var body: some View {
        Toggle(title, isOn: $isOn)
            .onChange(of: isOn) {
                onChange()
            }
    }
}

/// Generic enum picker. Works on any `RawRepresentable` enum that is
/// `CaseIterable` and exposes a `displayName: String`. Used for every
/// popup parameter.
struct EnumPicker<EnumType: CaseIterable & Identifiable & Hashable>: View
    where EnumType.AllCases: RandomAccessCollection {

    let title: String
    @Binding var selection: EnumType
    var onChange: () -> Void = {}

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.callout)
            Spacer()
            Picker(title, selection: $selection) {
                ForEach(Array(EnumType.allCases)) { option in
                    Text(displayName(for: option))
                        .tag(option)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 180, alignment: .trailing)
            .onChange(of: selection) {
                onChange()
            }
        }
    }

    private func displayName(for option: EnumType) -> String {
        if let displayable = option as? any DisplayNamed {
            return displayable.displayName
        }
        return String(describing: option)
    }
}

/// Marker protocol so `EnumPicker` can pull a localised display name
/// from any of our parameter enums without having to special-case them
/// individually. Each enum gets a one-line conformance below.
protocol DisplayNamed {
    var displayName: String { get }
}

extension ScreenColor: DisplayNamed, Identifiable { public var id: Int { rawValue } }
extension QualityMode: DisplayNamed, Identifiable { public var id: Int { rawValue } }
extension OutputMode: DisplayNamed, Identifiable { public var id: Int { rawValue } }
extension SpillMethod: DisplayNamed, Identifiable { public var id: Int { rawValue } }
extension UpscaleMethod: DisplayNamed, Identifiable { public var id: Int { rawValue } }
