# Release Notes

### 1.0.0 (Build 10)

**🎉 Released:**
- 4th May 2026

**🔨 Improvements:**
- We now have a new icon, designed by the amazing [Matthew Skiles](http://matthewskiles.com)!
- Now runs on macOS Sonoma 14.6 or later (on Apple Silicon only).
- Increased the default window size for the Standalone Editor.
- Added a built-in default **Custom Image Background** of a castle for easier testing out-of-the-box.
- Combined two MLX render stages for slightly faster performance.

---

### 1.0.0 (Build 9)

**🎉 Released:**
- 1st May 2026

**🔨 Improvements:**
- **Screen Colour: Blue** now uses Corridor Digital's dedicated blue-screen MLX model (`CorridorKeyBlue v1.0`). Previously the renderer rotated blue footage into the green domain so the green-only model could process it. The new path feeds blue source straight into a model trained on actual blue-screen footage and produces noticeably cleaner mattes on hair, motion blur, and translucent fabric.
- Despill, edge decontaminate, and the chroma-prior alpha hint are now screen-colour aware throughout — every post-process stage operates on the user's chosen screen colour rather than always assuming green.
- Green-screen behaviour and quality are unchanged. The green model and pipeline continue to ship as the default.
- **Hint: Apple Vision Framework** now layers the Vision subject mask on top of the chroma prior. Vision's foreground-instance detector segments people, animals, and salient subjects but ignores generic foreground objects. On shots with props (a sword, an instrument, a piece of set dressing) those would previously fall out of the matte. The combined hint keeps Vision's strong subject signal where it fires and falls back to chroma everywhere else, so foreground props in front of the screen stay in the key.
- Every parameter row in the **Standalone Editor** inspector now shows a **↺ Reset to Default** button next to the value when it differs from the factory default, so a single click restores the factory default value without the user having to remember the original number.
- **Despill Strength** now also pulls matte alpha toward zero in pixels strongly biased toward the screen colour, matching the original CorridorKey reference's "premultiply despilled foreground by alpha" output convention. Previously, model-error pixels in the screen background showed up as a coloured halo on the composite — light blue at strength 0, greenish at strength 1 — because the despilled colour stayed bound to a non-zero alpha. Pushing the slider up now does what the name implies: the spill goes away, transparently.
- The **Standalone Editor** now auto-detects the **Screen Colour** from the first frame on import and sets the picker accordingly, so a blue-screen plate keys correctly without the user having to flip the popup first. The user's manual choice still takes precedence after import — the auto-detect only runs once per clip load.
- The Standalone Editor now remembers the last used **Quality**, **Hint**, and **Upscale Method** between sessions, so a user who's tuned their workflow doesn't have to re-pick the same three popups every time the editor opens. Per-clip parameters (sliders, screen colour) still start fresh — the Reset to Default affordance covers factory recovery for those.
- The Standalone Editor now also remembers the last-picked **Player Background** (checkerboard / white / black / yellow / red / custom colour / custom image) between sessions, so the preview opens with the same backdrop the user left it on. The custom-image bookmark already survived restarts. The new persistence covers the picker's current case so the choice itself is no longer reset to checkerboard on every launch.
- The **Player Background** picker now shows a small coloured swatch next to each preset (white / black / yellow / red) so the rows are visually distinguishable, and the **Custom Colour…** row mirrors the user's current swatch so the menu shows what you'll actually get without opening the colour wheel first. The previous icons were all rendered with the system foreground tint and were indistinguishable from each other.
- Quitting the **Standalone Editor** while an analyse pass is in progress now drains MLX's GPU stream and every Metal command queue before the process exits. Without this, Debug builds with Metal API Validation enabled would trip a `notifyExternalReferencesNonZeroOnDealloc` assertion when global pipeline state objects released while a command buffer still referenced them. Release builds were not crashing, but the lifecycle was racy - both paths are now deterministic.
- **Despill Strength** range expanded from 0–1 to 0–5 in both the Final Cut Pro inspector and the Standalone Editor. Pushing past 1 over-corrects the chroma in problem regions (heavy reflection on hair, dense motion-blur edges) and the new spill-alpha attenuation pulls the matte to zero in spill regions, so the over-correction lands as transparency instead of a garish anti-screen tint. The default stays at 0.5 - existing projects open identically.

---

### 1.0.0 (Build 8)

**🎉 Released:**
- 30th April 2026

**🐞 Bug Fix:**
- Fixed a crash on quit, that could happen if you try quit the application whilst the MLX pipeline is still processing something.

---

### 1.0.0 (Build 7)

**🎉 Released:**
- 30th April 2026

**🔨 Improvements:**
- Added the ability to load custom background colours and custom images into the Standalone Editor.
- Cleaned up the launch screen on the Standalone Editor.
- Opening Final Cut Pro from the Standalone Editor no longer quits the Standalone Editor.
- **Spill Method** now defaults to **Ultra (Chroma Project)**.
- Cleaned up the header section in the Final Cut Pro Effects Inspector to better match the Standalone Editor.
- Improvements to the on-screen controls in Final Cut Pro.
- Improvements to how we handle the colour space in Final Cut Pro, which improves the Apple Vision Framework Hint pipeline.

---

### 1.0.0 (Build 6)

**🎉 Released:**
- 27th April 2026

**🔨 Improvements:**
- Added HEVC exports. Thanks for suggesting Alex "4D" Gollner!
- General MLX and Vision Framework performance and stability improvements.
- Better GPU caching and pre-loading.
- Improvements to the Export window user interface in the Standalone Editor.
- Improvements to the playback controls and playback performance in the Standalone Editor.

---

### 1.0.0 (Build 5)

**🎉 Released:**
- 27th April 2026

**🔨 Improvements:**
- **Corridor Key Toolbox** has been renamed to **CorridorKey by LateNite**.

---

### 1.0.0 (Build 4)

**🎉 Released:**
- 26th April 2026

**🔨 Improvements:**
- The **CorridorKey by LateNite** application in your `/Applications` folder now has a **Standalone Editor** built-in, allowing you to export without launching Final Cut Pro.

---

### 1.0.0 (Build 3)

**🎉 Released:**
- 26th April 2026

**🔨 Improvements:**
- We now default to **Recommended** Quality (which picks best Quality setting based on your GPU memory).
- Added on-screen controls with new **Subject Position** parameter.
- Added **Temporal Stability** (matte flicker reduction) parameters.
- Added **Auto Subject Hint** - which uses Apple's Vision Framework to use machine learning object detection for the hint matte hint.
- Added new **Hint (Diagnostic)** Output, so you can see what the Hint is actually doing.
- Added **Edge Spill** Parameters.
- General improvements to memory usage and performance. It SHOULD be faster, more stable and better quality.

---

### 1.0.0 (Build 2)

**🎉 Released:**
- 24th April 2026

**🔨 Improvements:**
- This release is all about Performance Improvements & Bug Fixes.
- Added support for Zero-copy MLX I/O. We no longer use the CPU - it's all GPU. Yay! Saves roughly ~35ms / ~285MB per frame.
- Fixed some colour space bugs.
- Implemented some missing features (i.e. sliders that didn't actually do anything yet).
- Focussed on making it faster and better.
- Changed the default values and range of the Despeckle Size parameter.

---

### 1.0.0 (Build 1)

**🎉 Released:**
- 23rd April 2026

This is the first public beta release of **CorridorKey by LateNite**. Woohoo! 🥳
