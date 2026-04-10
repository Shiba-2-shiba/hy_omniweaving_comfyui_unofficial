# hy-omniweaving-comfyui

Custom-node extraction lane for HY-OmniWeaving support on top of ComfyUI.

Current scope:

- dedicated dual text-encoder loader for OmniWeaving's fine-tuned Qwen2.5-VL plus ByT5
- dedicated VAE loader path for OmniWeaving-style `AutoencoderKLConv3D` checkpoints
- dedicated Redux-backed vision encode path for SigLIP image encoder + Redux image embedder checkpoints
- `i2v` / `t2v` workflow support
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

- full task parity beyond `i2v` / `t2v`
- full resolution-bucket parity
- UI/workflow backward compatibility

Current status:

- This repo is currently in its best-known refactor state so far for practical `i2v` / `t2v` use on top of stock ComfyUI inference.
- `t2v` now runs cleanly and produces sharp, stable output in the current validated workflow.
- `i2v` now runs end-to-end without the previous runtime crash and no longer shows the earlier progressive collapse from the first frame in the current validated workflow.
- `HY OmniWeaving Text Encoder Loader` mirrors the intended HunyuanVideo 1.5 dual-loader flow: load the OmniWeaving fine-tuned Qwen checkpoint together with the ByT5 checkpoint as one CLIP output.
- `HY OmniWeaving VAE Loader` detects OmniWeaving-style `decoder.conv_in.conv.weight` checkpoints and instantiates a local `AutoencoderKLConv3D`-equivalent model from tracked config instead of relying on generic Comfy VAE fallback paths.
- `HY OmniWeaving Image Prep` is the blessed image preprocessing step for parity-sensitive `i2v` / `reference2v` style workflows.
- `HY OmniWeaving I2V Semantic Images` reproduces the original i2v VAE roundtrip so the same semantic first frame can feed CLIP-Vision and text-side multimodal input.
- `HY OmniWeaving Redux Vision Encode` loads a selected SigLIP image encoder checkpoint plus a selected Redux image embedder checkpoint from `models/clip_vision`, uses bundled tracked configs for the fixed local pair, and falls back to state-dict shape inference when alternate weights do not match those configs.
- Runtime-patch reduction has started: deepstack `mm_in` is attached loader-side after model load, `all_stack_text_states` is injected by patching `extra_conds` per loaded model instance, text-encoder deepstack/setclip support is applied per loaded clip instance, and early-block deepstack injection is attached through a per-model `DIFFUSION_MODEL` wrapper.
- Remaining global compatibility patches are applied lazily when the relevant loader/node path is used, not at package import time.

Known limitations:

- The tested public checkpoints still have `mm_in.linear_2.weight == 0` and `mm_in.linear_2.bias == 0`, so deepstack transport is structurally wired but the connector output is numerically inactive.
- `i2v` / `t2v` logs still show a small `cond_tokens` vs `deepstack_tokens` mismatch after crop/setclip. This is a remaining parity-quality issue, not a current execution blocker.
- Prompt-following and motion-heavy scenes can still diverge from the original runtime, especially where OmniWeaving-specific deepstack behavior would matter.

Recommended blessed paths:

- `i2v`
  1. reference image
  2. `HY OmniWeaving Image Prep`
  3. `HY OmniWeaving I2V Semantic Images`
  4. `HY OmniWeaving Redux Vision Encode`
  5. `HY OmniWeaving Text Encode`
  6. `HY OmniWeaving Conditioning`
  7. stock ComfyUI sampler / scheduler / CFG

- `t2v`
  1. `HY OmniWeaving Text Encode` with `task=t2v`
  2. `HY OmniWeaving Conditioning` with `task=t2v`
  3. stock ComfyUI sampler / scheduler / CFG

Debugging notes:

- In the current validated `i2v` path, text-side multimodal input still prefers explicit `semantic_images` when they are connected, even if `HY OmniWeaving Redux Vision Encode` also populates `clip_vision_output.mm_projected`.
- `mm_projected` being `None` is no longer expected in the Redux-backed `i2v` workflow, but it can still appear in fallback/legacy CLIP-Vision paths.
- `unet unexpected: ['mm_in...']` is currently a diagnostic signal, not by itself a failure, because the stock loader ignores OmniWeaving-only `mm_in.*` keys and the custom loader re-attaches them afterward.
- The previous `i2v` runtime crash caused by an invalid `ref_latent` payload has been removed from the current conditioning path.
