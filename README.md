# hy-omniweaving-comfyui

Custom-node extraction lane for HY-OmniWeaving support on top of stock
ComfyUI.

## Scope

- dedicated dual text-encoder loader for the OmniWeaving-tuned Qwen2.5-VL
  checkpoint plus ByT5
- dedicated VAE path for OmniWeaving-style
  `AutoencoderKLConv3D` checkpoints
- dedicated Redux-backed vision encode path for local SigLIP image encoder +
  Redux image embedder checkpoints
- `i2v` / `t2v` workflow support
- HY-OmniWeaving-oriented text encoding and conditioning
- optional think modes: `legacy_rewrite`, `merge_hidden`
- deepstack / setclip compatibility
- Omni-style image preparation and i2v semantic-image roundtrip

## Boundary

- Keep **sampling / scheduler / CFG / steps / model loading / VRAM
  management** on stock ComfyUI nodes.
- Keep only **OmniWeaving-specific text semantics, image preparation, task
  conditioning, and compatibility hooks** in this package.
- Set `HY_OMNIWEAVING_DEBUG=1` only when you need loader / text-encoder /
  mask / UNet diagnostics.

## Current status

- `i2v` and `t2v` both run end-to-end in the current validated workflow.
- The earlier `i2v` crashes from invalid `ref_latent` payloads and
  text-mask-length mismatches are gone in the current path.
- `HY OmniWeaving Text Encoder Loader` mirrors the intended dual-loader flow:
  OmniWeaving Qwen + ByT5 as one CLIP output.
- `HY OmniWeaving VAE Loader` detects OmniWeaving-style
  `decoder.conv_in.conv.weight` layouts and instantiates the local
  `AutoencoderKLConv3D`-equivalent model from tracked config.
- `HY OmniWeaving Image Prep` remains the blessed image-entry point for
  parity-sensitive `i2v` / `reference2v` style runs.
- `HY OmniWeaving I2V Semantic Images` reproduces the VAE roundtrip used to
  derive the semantic first frame for text-side multimodal input.
- `HY OmniWeaving Redux Vision Encode` populates `mm_projected` and exposes
  the local SigLIP + Redux path through `clip_vision_output`.

## Runtime patch status

- deepstack `mm_in` is attached loader-side after model load
- `all_stack_text_states` is injected through instance-local `extra_conds`
- text-encoder deepstack / setclip support is applied per loaded clip instance
- early-block deepstack injection is attached through a per-model
  `DIFFUSION_MODEL` wrapper
- `setclip` text-mask handling now works in the current validated `i2v` path:
  - the integrated text encoder still returns no `attention_mask`
  - the runtime patch reconstructs it from the processed Qwen branch
  - the final mask is expanded for `clip_vision_output` prefix tokens
  - oversized expanded masks are trimmed back to the active text length before
    `txt_in`
- ByT5 prefix expansion is supported in the final mask path, but the latest
  validated `i2v` logs still show `conditioning_byt5small_shape=None`
  because no quoted-text ByT5 route was active in that run

## Deepstack note

- The currently tested checkpoints are no longer showing the earlier
  `mm_in.linear_2 == 0` situation.
- Recent debug logs show non-zero `source_linear2_norm`,
  `attached_linear2_norm`, and `projected_norm`.
- In practice, the connector path is now numerically active in the tested run.
- Remaining divergence from the original runtime should currently be treated
  first as prompting / branch-content / implementation-style issues, not as
  proof that deepstack is inactive.

## Think-mode notes

- `legacy_rewrite`
  - closest to public-repo behavior
  - rewrites the prompt and re-encodes it once
  - currently the safer path when public-repo-style behavior matters more than
    hidden-state experimentation
- `merge_hidden`
  - keeps the original prompt branch
  - encodes the AR-generated continuation as the auxiliary branch
  - with `think_keep_tokens=0`, keeps the full generated branch by default
  - when an explicit cap is used, the retained segment is taken from the
    **front** of the generated branch, not the tail
  - remains a research path focused on motion gain vs prompt drift tradeoffs

## Current limitations

- `merge_hidden` is still the main quality-risk area:
  - the AR branch can over-describe static appearance or background
  - it can still weaken prompt fidelity even though the runtime is now stable
- exact processor/tokenization parity with the original repo is still not
  claimed
- the latest validated `i2v` logs do not show the earlier
  `cond_tokens` vs `deepstack_tokens` warning, but crop/setclip parity should
  still be treated as a refinement target
- ByT5 prefix behavior is now instrumented in the final mask path, but a real
  run with active `conditioning_byt5small` still needs explicit validation

## Recommended blessed paths

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

## Project notes

Detailed notes now live under [`doc/`](./doc/README.md):

- [`doc/status.md`](./doc/status.md)
- [`doc/thinking-modes.md`](./doc/thinking-modes.md)
- [`doc/parity.md`](./doc/parity.md)

## Debugging notes

- In the current validated `i2v` path, text-side multimodal input still
  prefers explicit `semantic_images` when they are connected, even if
  Redux-backed `clip_vision_output` is also present.
- `mm_projected` being `None` is no longer expected in the Redux-backed path,
  but it can still appear in fallback / legacy CLIP-Vision routes.
- `unet unexpected: ['mm_in...']` is still a diagnostic signal only; the
  stock loader ignores `mm_in.*` and the custom loader re-attaches it.
- If you want to confirm the current mask path, inspect:
  - `attention_mask reconstructed ... source=qwen_branch`
  - `text encode final attention_mask expansion ...`
  - `txt_in mask alignment ... effective_mask_shape=...`
- If you want to confirm that deepstack is active, inspect:
  - `unet loader detected mm_in tensors ... source_linear2_norm=...`
  - `mm_in source_vs_attach ... attached_linear2_norm=...`
  - `diffusion wrapper fired ... projected_norm=...`
