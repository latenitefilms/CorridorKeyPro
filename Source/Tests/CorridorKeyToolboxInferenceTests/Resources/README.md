# Inference test fixtures

`MLXMemoryTests` runs real MLX inference outside Final Cut Pro to catch
the unbounded-cache regression that surfaced in v1.0.0 build 2 analyse
passes. It needs two artefacts at runtime:

1. **`corridorkey_mlx_bridge_512.mlxfn`** — the smallest MLX bridge from
   the production resources folder. It's 288 MB and gitignored, so
   re-populate it after a fresh checkout:

   ```bash
   cp "Source/CorridorKeyToolbox/Plugin/Resources/MLX Models/corridorkey_mlx_bridge_512.mlxfn" \
      "Source/Tests/CorridorKeyToolboxInferenceTests/Resources/"
   ```

2. **`mlx-swift_Cmlx` Metal library** — `swift test` doesn't run the
   Xcode build plugin that compiles MLX's own kernels, so we re-use the
   one Xcode produces. After at least one Xcode build of the plugin,
   stage it into the SPM debug dir:

   ```bash
   SOURCE=$(find ~/Library/Developer/Xcode/DerivedData/CorridorKeyToolbox-*/Build/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources \
       -name default.metallib | head -1)
   DEST="Source/.build/arm64-apple-macosx/debug/CorridorKeyToolboxLogicPackageTests.xctest/Contents/MacOS"
   mkdir -p "$DEST"
   cp "$SOURCE" "$DEST/mlx.metallib"
   cp "$SOURCE" "$DEST/default.metallib"
   ```

   (The MLX runtime walks several colocated paths looking for the kernel
   library; staging both names covers them all.)

Then run:

```bash
cd Source && swift test --filter MLXMemory
```

Expected output: three tests pass, with a print line confirming MLX cache
peaks around **256 MB** (cap from `MLXKeyingEngine.applyMemoryLimitsOnce`).
Pre-fix this number was ~4 GB at the 512 rung and ~70 GB at 2048.
