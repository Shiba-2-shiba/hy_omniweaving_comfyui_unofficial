# HY OmniWeaving ComfyUI Status

This note summarizes the current best-known state of the refactor and the
conclusions that are stable enough to build on.

## Goal

Primary goal:

- achieve `i2v` / `t2v` quality in ComfyUI that is as close as possible to the
  original OmniWeaving repository

Architectural goal:

- keep **sampling / scheduler / CFG / model loading / VRAM management** inside
  stock ComfyUI
- move only **OmniWeaving-specific semantics and compatibility logic** into this
  custom node

## Current best-known state

- `t2v` runs successfully in the current validated workflow
- `i2v` runs successfully in the current validated workflow
- the earlier `i2v` crashes from invalid `ref_latent` payloads are removed
- the later `i2v` crashes from text-mask-length mismatches are also removed in
  the current validated path

This does **not** mean full original-repo parity has been reached. It means the
current custom-node direction is functionally usable and materially better than
earlier refactor states.

## What is now working

### 1. Boundary direction is correct

- ComfyUI owns:
  - sampler
  - scheduler
  - CFG
  - step count
  - model loading / offload / dtype handling
- custom node owns:
  - OmniWeaving text prompt semantics
  - OmniWeaving image preparation
  - OmniWeaving task conditioning
  - OmniWeaving-specific deepstack compatibility

### 2. Blessed `i2v` path

1. reference image
2. `HY OmniWeaving Image Prep`
3. `HY OmniWeaving I2V Semantic Images`
4. `HY OmniWeaving Redux Vision Encode`
5. `HY OmniWeaving Text Encode`
6. `HY OmniWeaving Conditioning`
7. stock ComfyUI sampler / scheduler / CFG

Current confirmed details:

- text-side multimodal input is intentionally driven by `semantic_images`
- Redux vision fills `clip_vision_output.mm_projected`, but the current blessed
  text path still prefers explicit `semantic_images`

### 3. Blessed `t2v` path

1. `HY OmniWeaving Text Encode` with `task=t2v`
2. `HY OmniWeaving Conditioning` with `task=t2v`
3. stock ComfyUI sampler / scheduler / CFG

Current confirmed details:

- the explicit zero-conditioning path is working as intended
- `t2v` does not depend on accidental visual-input leakage

### 4. VAE compatibility is stable

- Omni layout detection succeeds
- expected 3D VAE tensor shapes are recognized
- the local `AutoencoderKLConv3D`-equivalent path is stable in validated runs

### 5. Deepstack transport is structurally fixed

- `all_stack_text_states` preserves the layer axis
- deepstack attachment and diffusion-time injection are instance-local
- current tested logs show non-zero `source_linear2_norm`,
  `attached_linear2_norm`, and `projected_norm`

Current assessment:

- deepstack transport is structurally fixed
- in the tested run, the connector path is numerically active

### 6. Current setclip / text-mask path is viable

Observed:

- the integrated ComfyUI text encoder still returns no integrated
  `attention_mask`
- the runtime patch reconstructs `attention_mask` from the processed Qwen branch
- the final text mask is expanded for `clip_vision_output` prefix tokens
- oversized expanded masks are trimmed back to the active text length before
  `txt_in`

Current assessment:

- this is not byte-for-byte original-repo behavior
- it is stable enough for the current validated `i2v` path

## Remaining known issues

### A. `merge_hidden` branch content can still overpower the user prompt

Observed:

- the current full-branch `merge_hidden` path keeps the entire generated branch
  by default
- real debug runs still show auxiliary text that over-specifies static
  appearance or background
- quality drift can still appear even though runtime now completes

Current assessment:

- runtime stability is no longer the main issue here
- branch content quality is the main open risk

### B. Exact crop / setclip parity is still not claimed

Observed:

- the latest validated `i2v` logs no longer show the earlier token-length
  warning
- the current path still uses custom runtime-patch reconstruction and
  expansion for integrated text masks

Current assessment:

- not a current execution blocker
- still a valid parity-cleanup target

### C. ByT5-active final mask path is instrumented but still unconfirmed in a real run

Observed:

- the latest validated `i2v` logs show `conditioning_byt5small_shape=None`
- ByT5 prefix expansion is supported in the final mask path, but this run did
  not exercise it

Current assessment:

- not a current blocker
- still needs explicit validation in a real ByT5-active case

## Suggested next steps

### Priority 1

Keep the current blessed `i2v` / `t2v` workflows stable.

- do not reintroduce the invalid `ref_latent` path casually
- keep `semantic_images` explicit for `i2v`
- keep `HY OmniWeaving Image Prep` as the blessed image-entry point

### Priority 2

Investigate the remaining quality gap in `merge_hidden`.

- inspect whether the AR-generated branch over-describes static appearance,
  pose, or background
- evaluate whether full-branch retention should remain the default after more
  visual checks
- if prompt drift remains, trim or filter static-description-heavy sentences

### Priority 3

Validate the currently instrumented ByT5 prefix path in a real run.

- run a case that actually produces `conditioning_byt5small`
- confirm that final mask-prefix expansion remains correct when ByT5 is active

## Keep doing

- keep debug logs available behind `HY_OMNIWEAVING_DEBUG=1`
- keep instance-local adapters instead of returning to global monkey patches
- keep the blessed `i2v` path centered on prepared image -> semantic image ->
  text/vision conditioning
- keep the blessed `t2v` path text-only
- keep the current `setclip` text-mask path observable through reconstruction /
  expansion / `txt_in` alignment logs

## Avoid doing

- do not suppress `mm_in` unexpected-key warnings just to quiet logs
- do not revert to `clip_vision_output`-only text-side visual input for `i2v`
- do not reintroduce `ref_latent` without verifying the exact current ComfyUI
  runtime channel contract
- do not assume the current `merge_hidden` branch content is safe just because
  the runtime now completes
