# System Requirements

**CorridorKey by LateNite** requires **Final Cut Pro 11** or later on **macOS 26 Tahoe**.

It uses **Apple MLX** and requires a modern **Apple Silicon** Mac with as much RAM as you can throw it.

It does **NOT** work on Intel Mac's.

You can find Final Cut Pro's system requirements [here](https://support.apple.com/en-us/111903).

---

## CorridorKey Requirements

For reference, the official CorridorKey GitHub repository has the following to say about Hardware Requirements:

> This project was designed and built on a Linux workstation (Puget Systems PC) equipped with an NVIDIA RTX Pro 6000 with 96GB of VRAM. The community is ACTIVELY optimizing it for consumer GPUS.
>
> The most recent build should work on computers with 6-8 gig of VRAM, and it can run on most M1+ Mac systems with unified memory. Yes, it might even work on your old Macbook pro. Let us know on the Discord!
>
> *   **Windows Users (NVIDIA):** To run GPU acceleration natively on Windows, your system MUST have NVIDIA drivers that support **CUDA 12.8 or higher** installed. If your drivers only support older CUDA versions, the installer will likely fallback to the CPU.
> *   **AMD GPU Users (ROCm):** AMD Radeon RX 7000 series (RDNA3) and RX 9000 series (RDNA4) are supported via ROCm on **Linux**. Windows ROCm support is experimental (torch.compile is not yet functional). See the [AMD ROCm Setup](#amd-rocm-setup) section below.
> *   **GVM (Optional):** Requires approximately **80 GB of VRAM** and utilizes massive Stable Video Diffusion models.
> *   **VideoMaMa (Optional):** Natively requires a massive chunk of VRAM as well (originally 80GB+). While the community has tweaked the architecture to run at less than 24GB, those extreme memory optimizations have not yet been fully implemented in this repository.
> *   **BiRefNet (Optional):** Lightweight AlphaHint generator option.
>
> Because GVM and VideoMaMa have huge model file sizes and extreme hardware requirements, installing their modules is completely optional. You can always provide your own Alpha Hints generated from your editing program, BiRefNet, or any other method. The better the AlphaHint, the better the result.