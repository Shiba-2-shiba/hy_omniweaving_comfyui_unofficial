# HY OmniWeaving ComfyUI refactor status report

This note summarizes what has worked, what has failed, and what has been
learned during the current refactor pass.

Its purpose is to help future development avoid repeating already-settled
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

## High-level outcome so far

### What has improved

- text-side image input for `i2v` is now routed through images instead of
  relying on `CLIP_VISION_OUTPUT.mm_projected`
- deepstack tensor transport now preserves the **layer axis**
- deepstack model attachment is now **instance-local**, not global monkey patch
- text encoder patching is now **instance-local**
- diffusion-time deepstack injection is now **instance-local wrapper-based**
- VAE path is stable and correctly recognizes OmniWeaving's 3D VAE layout

### What has not improved enough yet

- prompt-following is still weaker than expected
- UNet-side OmniWeaving deepstack effect is still effectively zero because the
  model's `mm_in.linear_2` weights are zero in both fp8 and fp32 checkpoints

---

## Successful refactor steps

### 1. Boundaries were clarified correctly

This direction appears correct and should be preserved:

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

This reduced unnecessary reimplementation risk.

---

### 2. Blessed `i2v` image-prep path was introduced

Added:

- `HY OmniWeaving Image Prep`

Purpose:

- enforce original-style `Lanczos + center crop`
- let the same prepared image feed:
  - VAE conditioning
  - text-side multimodal input
  - CLIP-Vision conditioning path

This was a good change.

---

### 3. Prompt/template parity work was useful

The task-to-template mapping was aligned with original OmniWeaving
`prompt_mode` semantics:

- `t2v`
- `i2v`
- `reference2v`
- `interpolation`
- `editing`
- `tiv2v`

This helped reduce silent semantic drift in the text encoder path.

---

### 4. Global monkey patches were reduced successfully

The following global patches were removed in favor of instance-local behavior:

- model detection patch
- model base patch
- text encoder class-wide patch
- global HunyuanVideo `_forward` patch

Replaced by:

- per-model `mm_in` attach
- per-model `extra_conds` adapter
- per-model `DIFFUSION_MODEL` wrapper
- per-clip text encoder patch

This was a substantial architecture improvement.

---

### 5. Deepstack layer collapse bug was found and fixed

Important finding:

- text encode produced `all_stack_text_states` with shape like
  `(3, 1, tokens, 3584)`
- but the default conditioning path collapsed that to `(1, 1, tokens, 3584)`

Cause:

- using generic `CONDRegular` for a tensor whose first dimension is **layer**,
  not batch

Fix:

- introduced a dedicated deepstack conditioning carrier that preserves the
  layer axis and only expands/concats on the batch axis

This was an important real fix, not just logging.

---

## Failed or disproven assumptions

### 1. `CLIP_VISION_OUTPUT` alone was enough for text-side `i2v`

This turned out to be false for the tested workflow.

Observed:

- `clip_vision_output` was connected
- `mm_projected` was `None`
- `image_embeds=0`

Meaning:

- conditioning-side CLIP-Vision path existed
- but Qwen-side multimodal image input was still absent

Conclusion:

- `HYOmniWeavingTextEncode` must accept actual image input for parity-sensitive
  `i2v` runs

---

### 2. fp8 quantization was probably killing `mm_in.linear_2`

This was investigated and disproven.

Comparison result:

- `mm_in.linear_2.weight` is zero in the fp8 checkpoint
- `mm_in.linear_2.weight` is also zero in the fp32 checkpoint

Conclusion:

- this is **not** caused by fp8 quantization
- the zero output is already present in the original fp32 checkpoint being used

---

### 3. `mm_in` unexpected-key warnings were the main issue

This is not the best current interpretation.

The warning:

- `unet unexpected: ['mm_in.linear_1.bias', ...]`

is currently consistent with the chosen architecture:

- core ComfyUI load ignores OmniWeaving-only `mm_in.*`
- custom node re-attaches them after load

This warning should still be kept as a diagnostic signal, but it is no longer
the strongest root-cause candidate for quality issues.

---

## Confirmed current findings

### A. VAE path is in comparatively good shape

Observed:

- Omni layout detection succeeds
- expected 3D VAE tensor shapes are recognized

Current assessment:

- VAE is not the main blocker right now

---

### B. Text-side `i2v` image path is now structurally correct

Observed:

- `visual_source=reference_images`
- `visual_images=1`
- large text-side sequence length increase after the image-based path was wired

Current assessment:

- this change improved image faithfulness / stability

---

### C. Deepstack tensors are now preserved up to diffusion wrapper entry

Observed after custom COND fix:

- text encode output: `(3, 1, tokens, 3584)`
- wrapper input: `(3, 1, tokens, 3584)`
- patched blocks: `3`

Current assessment:

- structural deepstack transport is now working as intended

---

### D. `mm_in` projection is still numerically dead

Observed:

- `mm_in.linear_1.weight` is non-zero
- `mm_in.linear_2.weight` is all-zero
- `projected_norm` is effectively zero

Meaning:

- deepstack reaches the model
- but the learned connector path produces zero output

Current assessment:

- this is the strongest remaining explanation for weak prompt-following once
  the structural bugs were fixed

---

## Current best explanation of behavior

The current state appears to be:

1. image conditioning is better than before
2. text-side multimodal input is better than before
3. deepstack transport is structurally fixed
4. but the OmniWeaving deepstack connector itself is numerically inactive

Practical effect:

- image coherence improves
- prompt specificity remains weak

This matches the observed outputs so far.

---

## Recommended interpretation of remaining problem

At this stage, the main blocker is likely **model-weight-side**, not
pipeline-structure-side.

In particular:

- `mm_in.linear_2 == 0` in both tested checkpoints strongly suggests that
  OmniWeaving's expected deepstack effect is absent in the actual weights being
  used

This should be investigated separately from the ComfyUI integration layer.

---

## Suggested next steps

### Priority 1

Verify whether the original OmniWeaving runtime is expected to function with
the exact same checkpoint despite `mm_in.linear_2` being zero.

Questions:

- Is this intentional?
- Is another code path compensating for it?
- Is a different checkpoint supposed to be used?

### Priority 2

Continue quality comparison focusing on paths **other than** deepstack:

- prompt templates
- image preparation consistency
- conditioning-side vision path
- workflow wiring discipline

### Priority 3

If needed, add a repo note or experiment branch for:

- temporary deepstack bypass experiments
- replacement connector experiments
- direct comparison against original runtime outputs

These should be treated as experiments, not merged assumptions.

---

## Keep doing

- keep debug logs available behind `HY_OMNIWEAVING_DEBUG=1`
- keep `HY OmniWeaving Image Prep` in the blessed workflow
- keep text-side image input explicit for `i2v`
- keep instance-local adapters instead of returning to global monkey patches

---

## Avoid doing

- do not suppress `mm_in` unexpected-key warnings just to quiet logs
- do not revert to `clip_vision_output`-only text-side visual input for `i2v`
- do not assume fp8 is the reason `mm_in.linear_2` is zero
- do not reintroduce broad global patches unless a specific regression proves
  they are necessary
