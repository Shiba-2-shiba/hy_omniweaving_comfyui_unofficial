# hy-omniweaving-comfyui

Custom-node extraction lane for HY-OmniWeaving support on top of ComfyUI.

Current scope:

- dedicated dual text-encoder loader for OmniWeaving's fine-tuned Qwen2.5-VL plus ByT5
- dedicated VAE loader path for OmniWeaving-style `AutoencoderKLConv3D` checkpoints
- `i2v` first
- HY-OmniWeaving-oriented text encoding
- ByT5/visual-input parity guards
- optional `think` prompt expansion
- `deepstack` / `setclip` option plumbing
- Omni-style conditioning with Lanczos + center-crop image preparation

Recommended workflow boundary:

- keep **sampling / scheduler / CFG / steps / model loading / VRAM management** on stock ComfyUI nodes
- keep only **OmniWeaving-specific text semantics, image preparation, task conditioning, and compatibility hooks** in this package
- set `HY_OMNIWEAVING_DEBUG=1` temporarily when you need extra loader/attach diagnostics for text encoder / UNet / VAE status checks

Current non-goals:

- full task parity beyond `i2v`
- full resolution-bucket parity
- UI/workflow backward compatibility

Current status:

- This package is the new refactor target for HY-OmniWeaving-specific logic.
- Some lower-level parity support still lives in the current ComfyUI core checkout while the extraction is in progress.
- Future work should continue here rather than adding more Omni-specific behavior to generic ComfyUI Hunyuan modules.
- `runtime_patches.py` is the extraction hook for eventually moving the remaining core edits behind custom-node-owned monkey patches.
- `HY OmniWeaving Text Encoder Loader` now mirrors the intended HunyuanVideo 1.5 dual-loader flow: load the OmniWeaving fine-tuned Qwen checkpoint together with the ByT5 checkpoint as one CLIP output.
- `HY OmniWeaving VAE Loader` now detects OmniWeaving-style `decoder.conv_in.conv.weight` checkpoints and instantiates a local `AutoencoderKLConv3D`-equivalent model from tracked config instead of relying on generic Comfy VAE fallback paths.
- `HY OmniWeaving Image Prep` is the blessed image preprocessing step for parity-sensitive `i2v` / `reference2v` style workflows: prepare the same reference images once, then feed them both into CLIP-Vision encoding and the Omni conditioning node.
- `HY OmniWeaving I2V Semantic Images` now reproduces the original i2v VAE roundtrip so parity-sensitive workflows can feed the same semantic first frame into CLIP-Vision and text-side multimodal input.
- `prompt_template_parity.md` tracks the current task-to-`prompt_mode` mapping and the original `crop_start` metadata we still want to preserve as refactoring continues.
- Runtime-patch reduction has started: deepstack `mm_in` is now attached loader-side after model load, `all_stack_text_states` is injected by patching `extra_conds` per loaded model instance, text-encoder deepstack/setclip support is applied per loaded clip instance, and early-block deepstack injection is now attached through a per-model DIFFUSION_MODEL wrapper. We no longer need global model-detection/model-base/text-encoder/forward patches for those parts.
- Current DeepStacking assessment: the transport path is structurally wired and follows the original OmniWeaving injection shape, but the tested public checkpoints currently have `mm_in.linear_2.weight == 0` and `mm_in.linear_2.bias == 0`, so the connector output is numerically inactive even when attachment succeeds.
- Remaining global compatibility patches are now applied lazily when the relevant loader/node path is used, not at package import time.

Recommended blessed paths:

- `i2v`
  1. reference image
  2. `HY OmniWeaving Image Prep`
  3. `HY OmniWeaving I2V Semantic Images`
  4. stock ComfyUI CLIP-Vision encode
  5. `HY OmniWeaving Text Encode`
  6. `HY OmniWeaving Conditioning`
  7. stock ComfyUI sampler / scheduler / CFG

- `t2v`
  1. `HY OmniWeaving Text Encode` with `task=t2v`
  2. `HY OmniWeaving Conditioning` with `task=t2v`
  3. stock ComfyUI sampler / scheduler / CFG
