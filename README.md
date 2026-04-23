# Corridor Key Toolbox

**Corridor Key Toolbox** is an Motion Template that brings [CorridorKey](https://github.com/nikopueringer/CorridorKey)'s keying power to Final Cut Pro.

Niko explains:

> When you film something against a green screen, the edges of your subject inevitably blend with the green background. This creates pixels that are a mix of your subject's color and the green screen's color. Traditional keyers struggle to untangle these colors, forcing you to spend hours building complex edge mattes or manually rotoscoping. Even modern "AI Roto" solutions typically output a harsh binary mask, completely destroying the delicate, semi-transparent pixels needed for a realistic composite.
>
> I built CorridorKey to solve this *unmixing* problem.
>
> You input a raw green screen frame, and the neural network completely separates the foreground object from the green screen. For every single pixel, even the highly transparent ones like motion blur or out-of-focus edges, the model predicts the true, un-multiplied straight color of the foreground element, alongside a clean, linear alpha channel. It doesn't just guess what is opaque and what is transparent; it actively reconstructs the color of the foreground object as if the green screen was never there.
>
> No more fighting with garbage mattes or agonizing over "core" vs "edge" keys. Give CorridorKey a hint of what you want, and it separates the light for you.

---

## Targets

| Target                          | Product                                 | Role                                                                 |
| ------------------------------- | --------------------------------------- | -------------------------------------------------------------------- |
| `Corridor Key Toolbox`          | `Corridor Key Toolbox.app`              | Wrapper application required for App Store packaging and discovery.  |
| `Corridor Key Toolbox Renderer` | `Corridor Key Toolbox.pluginkit` (XPC)  | The actual FxPlug plug-in Final Cut Pro loads into its plug-in host. |

The wrapper embeds the XPC service inside its `Contents/PlugIns` directory, so a single `.app` bundle is the unit of distribution.

---

## Layout

```
Source/
├── CorridorKeyToolbox.xcodeproj
├── Configuration/
│   └── Shared.xcconfig             # Shared build settings
├── CorridorKeyToolbox/
│   ├── Plugin/
│   │   ├── main.swift              # XPC entry point
│   │   ├── CorridorKeyToolboxPlugIn.swift
│   │   ├── CorridorKeyToolboxPlugIn+*.swift (protocol slices)
│   │   ├── Parameters/             # Identifiers, enums, state, UI
│   │   ├── Render/
│   │   │   └── RenderPipeline.swift
│   │   ├── Inference/
│   │   │   ├── KeyingInferenceEngine.swift
│   │   │   ├── RoughMatteKeyingEngine.swift
│   │   │   ├── MLXKeyingEngine.swift
│   │   │   └── InferenceCoordinator.swift
│   │   ├── Metal/
│   │   │   ├── MetalDeviceCache.swift
│   │   │   └── CorridorKeyShaders.metal
│   │   ├── PostProcess/
│   │   │   └── ScreenColorEstimator.swift
│   │   ├── Shared/
│   │   │   ├── CorridorKeyShaderTypes.h
│   │   │   └── CorridorKeyToolbox-Bridging-Header.h
│   │   ├── Resources/en.lproj/InfoPlist.strings
│   │   ├── Info.plist
│   │   └── CorridorKeyToolbox.entitlements
│   └── WrapperApp/
│       ├── CorridorKeyToolboxApp.swift # SwiftUI app entry point
│       ├── WelcomeView.swift
│       ├── Info.plist
│       ├── WrapperApp.entitlements
│       └── en.lproj/InfoPlist.strings
```

---

## Architecture overview

1. **FxPlug integration** – `CorridorKeyToolboxPlugIn` conforms to both
   `FxTileableEffect` (render callbacks) and `FxAnalyzer` (pre-render analysis).
   Extensions split the class so each protocol responsibility lives in a
   dedicated file.
2. **Parameter layer** – `ParameterIdentifiers.swift` assigns every UI control
   a stable identifier. `CorridorKeyToolboxPlugIn+Parameters.swift` builds the
   inspector; `CorridorKeyToolboxPlugIn+PluginState.swift` reads those values into
   a `PluginStateData` snapshot that the renderer owns for the duration of a
   tile render.
3. **Render pipeline** – `RenderPipeline` orchestrates the per-frame Metal
   work in explicit stages: screen-colour rotation → hint extraction →
   normalise for inference → neural keying → upscale → despill → matte
   refinement → source passthrough → restore screen colour → compose. Every
   stage is a dedicated compute kernel so the GPU timeline is easy to profile.
4. **Inference abstraction** – `KeyingInferenceEngine` is the protocol the
   pipeline uses to acquire a matte. `MLXKeyingEngine` loads a pre-compiled
   `.mlxfn` bridge via mlx-swift and is the preferred path;
   `RoughMatteKeyingEngine` is the always-available CPU fallback used when no
   bridge file is bundled.
5. **Metal device cache** – `MetalDeviceCache` lazily compiles pipelines for
   every `(device, pixelFormat)` pair Final Cut Pro gives us, and hands out
   command queues from a small pool. One cache is shared across every plug-in
   instance living in the XPC service.

---

## Building

Open `CorridorKeyToolbox.xcodeproj` in Xcode and build the `Corridor Key Toolbox` scheme.

On first open Xcode will resolve the `mlx-swift` Swift package; this takes ~30 seconds over a cold network and is cached thereafter.

---

## Engineering conventions

- Swift 6 with complete strict-concurrency enabled.
- Only first-party Apple + `mlx-swift` dependencies.
- All pixel maths in Metal compute shaders; the CPU side stays on the coordination hot path.
- Parameter identifiers are never renumbered: add new controls, never move existing ones.

---

## Acknowledgements

Corridor Key Toolbox would NOT be possible without Niko's amazing [Corridor Key](https://github.com/nikopueringer/CorridorKey).

Niko's Corridor Key repo integrates several open-source modules for Alpha Hint generation.

We would like to also explicitly credit and thank the following research teams:

- **Generative Video Matting (GVM):** Developed by the Advanced Intelligent Machines (AIM) research team at Zhejiang University. The GVM code and models are heavily utilized in the `gvm_core` module. Their work is licensed under the [2-clause BSD License (BSD-2-Clause)](https://opensource.org/license/bsd-2-clause). You can find their source repository here: [aim-uofa/GVM](https://github.com/aim-uofa/GVM). Give them a star!
- **VideoMaMa:** Developed by the CVLAB at KAIST. The VideoMaMa architecture is utilized within the `VideoMaMaInferenceModule`. Their code is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/), and their specific foundation model checkpoints (`dino_projection_mlp.pth`, `unet/*`) are subject to the [Stability AI Community License](https://stability.ai/license). You can find their source repository here: [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa). Give them a star!

By using these optional modules, you agree to abide by their respective Non-Commercial licenses. Please review their repositories for full terms.
