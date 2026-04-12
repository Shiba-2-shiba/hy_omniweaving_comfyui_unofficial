# hy-omniweaving-comfyui

Custom-node extraction lane for HY-OmniWeaving support on top of ComfyUI.

Current scope:

- dedicated dual text-encoder loader for OmniWeaving's fine-tuned Qwen2.5-VL plus ByT5
- dedicated VAE loader path for OmniWeaving-style `AutoencoderKLConv3D` checkpoints
- dedicated Redux-backed vision encode path for SigLIP image encoder + Redux image embedder checkpoints
- `i2v` / `t2v` workflow support
- HY-OmniWeaving-oriented text encoding
- ByT5/visual-input parity guards
- optional `think` modes (`legacy_rewrite`, `merge_hidden`)
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
- `legacy_rewrite` is currently the closest path to the public OmniWeaving inference implementation: AR prompt enhancement rewrites the prompt and then re-encodes it once.
- `merge_hidden` is no longer the earlier deterministic suffix approximation. It now:
  1. keeps the original prompt as the base branch
  2. runs AR prompt enhancement
  3. uses the AR `generated_text` as the auxiliary branch
  4. merges the original and enhanced hidden states
- `merge_hidden` now uses a dedicated AR request prompt that tries to focus on motion / temporal progression while suppressing unnecessary static background and appearance restatement.

Known limitations:

- `i2v` / `t2v` logs still show a small `cond_tokens` vs `deepstack_tokens` mismatch after crop/setclip. This is a remaining parity-quality issue, not a current execution blocker.
- Prompt-following and motion-heavy scenes can still diverge from the original runtime, especially where OmniWeaving-specific deepstack behavior would matter.
- `merge_hidden` is still experimental in quality behavior. Even with the AR-based auxiliary branch, early-frame drift can appear when the generated continuation over-describes static appearance or background details.

Current deepstack note:

- The currently tested checkpoints are no longer showing the earlier `mm_in.linear_2 == 0` situation.
- Recent debug logs show non-zero `source_linear2_norm` / `attached_linear2_norm` values and a non-zero projected norm in the diffusion wrapper.
- In practice, this means deepstack transport is not just structurally wired anymore; the connector path is now numerically active.
- Remaining quality issues should therefore be treated as behavior / guidance problems first, not as proof that `mm_in` is still inactive.

Think-mode notes:

- `legacy_rewrite`
  - closest to public-repo behavior
  - rewrites the prompt and re-encodes it once
  - currently the safer choice when parity with the public inference path matters more than paper-style hidden-state experimentation
- `merge_hidden`
  - keeps the original prompt branch
  - uses AR-generated continuation text as the auxiliary branch
  - with `think_keep_tokens=0`, keeps the full generated branch by default
  - explicit positive `think_keep_tokens` values still cap the auxiliary branch length
  - intended as the paper-closer research path, not the official-style path

Debugging think modes:

- Set `HY_OMNIWEAVING_DEBUG=1` to inspect:
  - `think rewrite generated_text`
  - `think rewrite rewritten_text`
  - `think merge enhanced_prompt`
  - `think merge generated_branch_text`
  - merge-time `cond` / `all_stack_text_states` / `attention_mask` shapes
- The main project notes for current think-mode behavior are in:
  - `thinking_mode_status.md`
  - `legacy_rewrite_lm_head_fix_notes.md`

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

Current tracked workflow:

- `workflow/workflow_i2v.json` is the tracked `i2v` workflow snapshot that has been updated alongside the current think-mode experiments.

Debugging notes:

- In the current validated `i2v` path, text-side multimodal input still prefers explicit `semantic_images` when they are connected, even if `HY OmniWeaving Redux Vision Encode` also populates `clip_vision_output.mm_projected`.
- `mm_projected` being `None` is no longer expected in the Redux-backed `i2v` workflow, but it can still appear in fallback/legacy CLIP-Vision paths.
- `unet unexpected: ['mm_in...']` is currently a diagnostic signal, not by itself a failure, because the stock loader ignores OmniWeaving-only `mm_in.*` keys and the custom loader re-attaches them afterward.
- If you want to confirm that deepstack is active in a real run, inspect:
  - `unet loader detected mm_in tensors ... source_linear2_norm=...`
  - `mm_in source_vs_attach ... attached_linear2_norm=...`
  - `diffusion wrapper fired ... projected_norm=...`
- The previous `i2v` runtime crash caused by an invalid `ref_latent` payload has been removed from the current conditioning path.
