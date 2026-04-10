# HY OmniWeaving ComfyUI refactor status report

This note summarizes the current best-known state of the refactor and records
which conclusions are now stable enough to build on.

Its purpose is to help future work avoid repeating already-settled
investigation loops.

---

## Goal

Primary goal:

- achieve `i2v` / `t2v` quality in ComfyUI that is as close as possible to the
  original OmniWeaving repository

Architectural goal:

- keep **sampling / scheduler / CFG / model loading / VRAM management** inside
  stock ComfyUI
- move only **OmniWeaving-specific semantics and compatibility logic** into this
  custom node

---

## Current best-known state

At the time of this note, this repo is in its best practical state so far:

- `t2v` runs successfully and produces sharp, stable output in the currently
  validated workflow
- `i2v` runs successfully in the currently validated workflow
- the earlier `i2v` failure mode where frames progressively collapsed from the
  first frame is no longer reproduced in the current validated workflow
- the earlier `i2v` runtime crash caused by an invalid `ref_latent` payload has
  been removed from the active conditioning path

This does **not** mean full original-repo parity has been reached. It means the
current custom-node direction is now functionally usable and materially better
than earlier refactor states.

---

## What is now working

### 1. Boundary direction is correct

This direction should be preserved:

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

This continues to look like the right architecture.

### 2. The blessed `i2v` path is now viable

The current best-performing path is:

1. reference image
2. `HY OmniWeaving Image Prep`
3. `HY OmniWeaving I2V Semantic Images`
4. `HY OmniWeaving Redux Vision Encode`
5. `HY OmniWeaving Text Encode`
6. `HY OmniWeaving Conditioning`
7. stock ComfyUI sampler / scheduler / CFG

Important confirmed detail:

- `i2v` text-side multimodal input is now intentionally driven by
  `semantic_images`
- the Redux vision node now fills `clip_vision_output.mm_projected`, but the
  current blessed text path still prefers explicit `semantic_images` when
  connected
- the Redux loader uses tracked in-repo configs for the fixed local encoder /
  embedder pair and falls back to state-dict shape inference when alternate
  weights do not match those configs

### 3. The blessed `t2v` path is also viable

The current best-performing path is:

1. `HY OmniWeaving Text Encode` with `task=t2v`
2. `HY OmniWeaving Conditioning` with `task=t2v`
3. stock ComfyUI sampler / scheduler / CFG

Important confirmed detail:

- the explicit zero-conditioning path for `t2v` is now working as intended
- `t2v` no longer depends on accidental visual-input leakage

### 4. VAE compatibility is in good shape

Observed:

- Omni layout detection succeeds
- expected 3D VAE tensor shapes are recognized
- the local `AutoencoderKLConv3D`-equivalent path is stable in the current
  validated runs

Current assessment:

- VAE compatibility is no longer the main blocker

### 5. Deepstack transport structure is fixed

Observed:

- `all_stack_text_states` now preserves the layer axis
- deepstack conditioning is no longer collapsed onto a generic batch-like path
- deepstack attachment and diffusion-time injection are now instance-local

Current assessment:

- structural deepstack transport is working
- this was a real fix, not just added logging

---

## What was fixed recently

### `i2v` runtime crash from invalid `ref_latent`

The previous `i2v` crash was caused by passing a raw semantic latent through
`ref_latent` even though the active ComfyUI HunyuanVideo runtime expected a
different channel contract at `img_in`.

Practical effect:

- sampling crashed before generation could proceed

Resolution:

- the invalid `ref_latent` payload was removed from the current `i2v`
  conditioning path
- `i2v` now proceeds using the working `concat_latent_image`,
  `concat_mask`, and `guiding_frame_index` path

Current assessment:

- this was a runtime-contract bug in the integration layer
- this issue is now considered resolved unless a later ComfyUI runtime change
  reintroduces a valid `ref_latent` contract we intentionally support

---

## Failed or disproven assumptions

### 1. `CLIP_VISION_OUTPUT` alone was enough for text-side `i2v`

This was false for the tested workflow.

Observed:

- in the older stock CLIP-Vision path, `clip_vision_output` was connected
- `mm_projected` was `None`
- text-side image embeddings were absent

Conclusion:

- parity-sensitive `i2v` runs must provide actual image input to the text path
- `semantic_images` is currently the correct practical route
- populating `mm_projected` through the Redux vision node is still useful for
  conditioning transport and future text-side fallback paths

### 2. fp8 quantization was probably killing `mm_in.linear_2`

This was investigated and disproven.

Observed:

- `mm_in.linear_2.weight` is zero in the tested fp8 checkpoint
- `mm_in.linear_2.weight` is also zero in the tested fp32 checkpoint

Conclusion:

- this is not caused by fp8 quantization
- the current public weights themselves are the issue

### 3. `mm_in` unexpected-key warnings were the main issue

This is not the best current interpretation.

Observed:

- stock ComfyUI load ignores OmniWeaving-only `mm_in.*`
- the custom loader re-attaches them afterward

Conclusion:

- keep the warning as a diagnostic signal
- do not treat it as the primary explanation for current behavior

---

## Remaining known issues

### A. Deepstack connector is still numerically inactive

Observed:

- `mm_in.linear_1.weight` is non-zero
- `mm_in.linear_2.weight` is all-zero
- `mm_in.linear_2.bias` is all-zero
- the runtime correctly warns that deepstack injection is numerically inactive

Meaning:

- transport is wired
- the learned connector currently produces no useful output

Current assessment:

- this is still the strongest explanation for weak OmniWeaving-specific
  deepstack steering
- this is currently a model-weight-side issue, not the main integration bug

### B. Small token-count mismatch remains after crop/setclip

Observed in current successful runs:

- `cond_tokens` and `deepstack_tokens` still differ by a small amount after
  crop/setclip

Meaning:

- the current path is good enough to run and produce clear output
- crop/setclip parity is still not exact

Current assessment:

- not a current execution blocker
- still a valid target for future parity cleanup

### C. Motion-heavy scenes can still flicker

Observed qualitatively in current successful runs:

- output is much clearer than before
- strong motion can still introduce some flicker

Current assessment:

- the major collapse bug is gone
- remaining issues are now in the quality/parity tier, not the execution tier

---

## Working interpretation

The current state appears to be:

1. `i2v` conditioning path is structurally viable
2. `t2v` conditioning path is structurally viable
3. VAE path is stable
4. deepstack transport is structurally fixed
5. deepstack connector weights are still numerically inactive
6. crop/setclip parity is close enough to run, but not exact

Practical effect:

- the repo is now usable for current `i2v` / `t2v` workflows
- the next layer of work is about parity refinement and quality, not basic
  execution recovery

---

## Suggested next steps

### Priority 1

Keep the current blessed `i2v` / `t2v` workflows stable.

- do not reintroduce the invalid `ref_latent` path casually
- keep `semantic_images` explicit for `i2v`
- keep `HY OmniWeaving Image Prep` as the blessed image-entry point

### Priority 2

Investigate the remaining crop/setclip token mismatch.

- compare current crop/setclip slicing against the original `prepare_input()`
  and encode behavior
- verify whether the remaining delta is expected metadata, special tokens, or a
  real alignment bug

### Priority 3

Track the model-provider fix for `mm_in.linear_2`.

- once a corrected checkpoint is available, re-evaluate prompt-following and
  motion-heavy scenes before making further architectural conclusions

---

## Keep doing

- keep debug logs available behind `HY_OMNIWEAVING_DEBUG=1`
- keep instance-local adapters instead of returning to global monkey patches
- keep the blessed `i2v` path centered on prepared image -> semantic image ->
  text/vision conditioning
- keep the blessed `t2v` path text-only

## Avoid doing

- do not suppress `mm_in` unexpected-key warnings just to quiet logs
- do not revert to `clip_vision_output`-only text-side visual input for `i2v`
- do not assume fp8 is the reason `mm_in.linear_2` is zero
- do not reintroduce `ref_latent` without verifying the exact current ComfyUI
  runtime channel contract
- do not treat the remaining crop/setclip mismatch as proof that the whole path
  is broken; it is now a refinement issue
