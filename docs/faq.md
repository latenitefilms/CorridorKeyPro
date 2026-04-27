# Frequently Asked Questions

## Why does this exist?

The official [CorridorKey](https://github.com/nikopueringer/CorridorKey) repository does work on Mac, however, it's written in Python and is not really optimised for the latest Apple Silicon hardware.

We wanted to take the amazing [MLX Models](https://huggingface.co/alexandrealvaro/corridorkey-models/tree/main) compiled by [Alexandre Alvaro](https://huggingface.co/alexandrealvaro) for [CorridorKey-Runtime](https://github.com/alexandremendoncaalvaro/CorridorKey-Runtime), and use them in something that's specifically built and optimised for Final Cut Pro and Mac's.

Using the models as a starting point, we wanted to make something that feels native to Mac.

Whilst we have other open source FxPlug effects on GitHub (such as [Gyroflow Toolbox](https://gyroflowtoolbox.fcp.cafe)), we also wanted to showcase this as a great complex FxPlug, that others can use as a reference - as it has everything, Metal, MLX, OSC, etc.

We hope this will be a great open source resource for people to really dive deep into FxPlug.

---

## Was this vibe coded?

Yes, and no. I have absolutely used Claude Code (`Opus 4.7 1M Max`) and Codex (`GPT-5.5 Extra High`) during the development of this product, in both Xcode and in the Desktop applications, however, a lot of the code and/or techniques has been taken from my other applications such as [BRAW Toolbox](https://brawtoolbox.fcp.cafe), [Gyroflow Toolbox](https://gyroflowtoolbox.fcp.cafe), [Metaburner](https://metaburner.fcp.cafe) and [Keyframe Toolbox](https://keyframetoolbox.fcp.cafe).

Because there's not a heap of information about FxPlug on the public Internet (aside from the stuff on [FCP Cafe](https://fcp.cafe/developers/fxplug/)), LLMs really struggle with FxPlug, and more nitty gritty Final Cut Pro specific stuff - so sometimes it's just easier and quicker to code things by hand.

Unlike a lot of vibe coded projects, where the creator never opens Xcode, this is all build and made in Xcode v26.4.1.

---

## Is the quality as good as CorridorKey?

To be honest, I'm really not sure - I haven't had a chance to properly test other CorridorKey projects yet.

My priority has been getting **CorridorKey by LateNite** working great in Final Cut Pro - but performance wise, and quality wise - and I haven't really had a chance to compare speed or quality.

If you've tested and compared, please let us know in the [FCP Cafe Discord](https://ltnt.tv/discord)!

---

## What other CorridorKey projects exist?

Here's some cool projects:

- [CorridorKey AE](https://github.com/iamjoshuadavies/corridorkey-ae) - Native Adobe After Effects plugin for advanced green-screen keying.
- [CorridorKey by blace.ai](https://aescripts.com/corridorkey-by-blace-ai/) - High rendering performance (2x-4x faster rendering times) and completely C++ based for maximum stability.
- [CorridorKey Engine](https://github.com/99oblivius/CorridorKey-Engine) - Async multi-GPU inference, optimization profiles, a JSON-RPC engine API, and a Textual TUI.
- [corridorkey-mlx](https://github.com/cmoyates/corridorkey-mlx) - MLX inference port of CorridorKey for Apple Silicon.
- [CorridorKey-Runtime](https://github.com/alexandremendoncaalvaro/CorridorKey-Runtime) - Native AI keying runtime and OFX plugin for DaVinci Resolve.
- [CorridorKeyOpenVINO](https://github.com/daniil-lyakhov/CorridorKeyOpenVINO) - Fast Inference on Intel Hardware with OpenVINO
- [EZ-CorridorKey](https://github.com/edenaion/EZ-CorridorKey) - a GUI for CorridorKey.
- [Iris — CorridorKey](https://github.com/DomCoganda/Iris-CorridorKey) - A native desktop frontend for CorridorKey by Corridor Digital.