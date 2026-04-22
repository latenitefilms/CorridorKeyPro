# Corridor Key Toolbox

An FxPlug4 plug-in for Final Cut Pro that ports [CorridorKey](https://github.com/nikopueringer/CorridorKey)'s AI green-screen keying to Apple Silicon.

---

## Targets

| Target                      | Product                             | Role                                                                 |
| --------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| `Corridor Key Toolbox`          | `Corridor Key Toolbox.app`              | Wrapper application required for App Store packaging and discovery.  |
| `Corridor Key Toolbox Renderer` | `Corridor Key Toolbox.pluginkit` (XPC)  | The actual FxPlug plug-in Final Cut Pro loads into its plug-in host. |

The wrapper embeds the XPC service inside its `Contents/PlugIns` directory, so a single `.app` bundle is the unit of distribution.

---

## Layout

```
Source/
├── CorridorKeyPro.xcodeproj
├── Configuration/
│   └── Shared.xcconfig             # Shared build settings
├── CorridorKeyPro/
│   ├── Plugin/
│   │   ├── main.swift              # XPC entry point
│   │   ├── CorridorKeyProPlugIn.swift
│   │   ├── CorridorKeyProPlugIn+*.swift (protocol slices)
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
│   │   │   └── CorridorKeyPro-Bridging-Header.h
│   │   ├── Resources/en.lproj/InfoPlist.strings
│   │   ├── Info.plist
│   │   └── CorridorKeyPro.entitlements
│   └── WrapperApp/
│       ├── CorridorKeyProApp.swift # SwiftUI app entry point
│       ├── WelcomeView.swift
│       ├── Info.plist
│       ├── WrapperApp.entitlements
│       └── en.lproj/InfoPlist.strings
```

---

## Architecture overview

1. **FxPlug integration** – `CorridorKeyProPlugIn` conforms to both
   `FxTileableEffect` (render callbacks) and `FxAnalyzer` (pre-render analysis).
   Extensions split the class so each protocol responsibility lives in a
   dedicated file.
2. **Parameter layer** – `ParameterIdentifiers.swift` assigns every UI control
   a stable identifier. `CorridorKeyProPlugIn+Parameters.swift` builds the
   inspector; `CorridorKeyProPlugIn+PluginState.swift` reads those values into
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

Open `CorridorKeyPro.xcodeproj` in Xcode and build the `Corridor Key Toolbox` scheme. On first open Xcode will resolve the `mlx-swift` Swift package; this takes ~30 seconds over a cold network and is cached thereafter.

---

## Engineering conventions

- Swift 6 with complete strict-concurrency enabled.
- Only first-party Apple + `mlx-swift` dependencies.
- All pixel maths in Metal compute shaders; the CPU side stays on the coordination hot path.
- Parameter identifiers are never renumbered: add new controls, never move existing ones.