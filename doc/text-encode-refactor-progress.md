# Text Encode Refactor Progress

Last updated: 2026-04-15

## Objective

Track the TextEncode refactor that moves the current custom node toward
source-like prepared-input ownership while keeping the existing single
finetuned `safetensors` model path.

## Current Status

Status: replanning after runtime mask-density and mask-order diagnosis

Current focus:

- define the refactor shape before code edits
- keep the currently validated `i2v` / `t2v` workflows stable
- prioritize text-path parity over latent-path experimentation

## Stable Facts

- current `i2v` and `t2v` workflows execute end-to-end
- current `i2v` text path uses `semantic_images` as the active visual source
- current `attention_mask` behavior is still reconstructed / expanded / trimmed
- exact source parity is not yet reached
- the refactor should keep the current single finetuned `safetensors` model

## Phase Tracker

| Phase | Name | Status | Notes |
|---|---|---|---|
| 0 | Behavior lock | Completed | targeted regression coverage now locks current `i2v` / `t2v` text-path assumptions |
| 1 | Prepared-input spec | Completed | local prepared-input spec and Stage 1 spec tests are in place |
| 2 | Non-video task migration | Completed | `t2v` / `i2v` / `reference2v` / `interpolation` now tokenize through prepared-input ownership on the main path |
| 3 | Crop/setclip ownership | Completed | runtime patch now accepts prepared metadata and prefers it for crop/setclip ownership |
| 4 | Late mask cleanup | Completed | final correction is now narrower and task-aware |
| 5 | Dense mask reduction | Pending | reconstructed mask appears full-coverage and may be redundant on the main path |
| 6 | Clip-vision mask ordering | Pending | `txt_in` still receives post-concat-length mask on the main `i2v` path |
| 7 | Remaining attention-mask / setclip gaps | Pending | integrated mask still missing and setclip is still heuristic-heavy |
| 8 | Video-frame support | Pending | required for source-like `editing` / `tiv2v` |

## Completed

- [x] Refactor direction narrowed to TextEncode parity first
- [x] Current `i2v` log and workflow were reviewed against the parity hypothesis
- [x] Decision made to keep the current single finetuned `safetensors` model
- [x] Initial spec / task / progress documents created
- [x] Stage 0 regression coverage audited for `i2v` / `t2v` text encode behavior
- [x] Added missing regression tests for `i2v` reference-image fallback and `t2v` crop/setclip option propagation
- [x] Targeted Stage 0 pytest run passed
- [x] Added `LocalPreparedInputSpec` and source-like prepared-input helpers to `nodes.py`
- [x] Added Stage 1 spec-generation tests for `t2v`, `i2v`, and `tiv2v` fallback/order behavior
- [x] Targeted Stage 1 pytest run passed
- [x] Migrated the main `t2v` / `i2v` / `reference2v` / `interpolation` text encode path to prepared-input spec ownership
- [x] Preserved the current `clip.tokenize()` backend and the current image-embed fallback path for non-blessed cases
- [x] Updated regression tests for source-like 560 thumbnail behavior and source-like text-only fallback behavior
- [x] Targeted Stage 2 pytest run passed
- [x] Added prepared metadata transport from `nodes.py` into `runtime_patches.py`
- [x] Made runtime crop ownership prefer prepared metadata over heuristic template detection
- [x] Marked text-only fallback setclip ownership explicitly in the runtime patch
- [x] Targeted Stage 3 pytest run passed
- [x] Skipped clip-vision attention-mask prefix expansion for `t2v`
- [x] Restricted `txt_in` mask trimming to intentional all-ones prefix-expansion cases
- [x] Added Stage 4 regression tests for narrowed final-mask behavior
- [x] Targeted Stage 4 pytest run passed
- [x] Added runtime diagnostics proving `forward_orig` currently receives a
  post-concat-length `txt_mask` on the main `i2v` path
- [x] Added `txt_in` prefix/suffix mask diagnostics to quantify the overhang
- [x] Added `setclip` token-position and token-alignment diagnostics for the
  main `i2v` path

## Next Actions

1. Reduce or omit dense no-op reconstructed masks on the main `i2v` path.
2. Then remove clip-vision mask pre-expansion before `txt_in`.
3. Then reduce `reconstructed_from_qwen_branch` reliance on the main path.
4. Then reduce heuristic `setclip_start=3` reliance.

## Blockers

Current blockers:

- none for planning

Expected implementation blockers:

- some current tests encode assumptions about the old custom-template internals
- `editing` and `tiv2v` do not yet expose text-side `video_frames` input
- final prefix alignment may remain partially architectural because
  `clip_vision_output` still enters outside the source text encoder

## Risks To Watch

- regressing the currently stable validated `i2v` path while chasing parity
- over-correcting late mask logic before prepared-input ownership is in place
- conflating think-mode quality issues with base text-path parity issues
- changing too many task types at once instead of staging `i2v` / `t2v` first

## Session Log

### 2026-04-13

- Reviewed current `i2v` workflow and debug log.
- Confirmed that current text-path behavior is still approximate:
  - `semantic_images` are the active text-side visual source
  - `attention_mask` is reconstructed from the Qwen branch
  - final mask prefix expansion and `txt_in` trim are still present
- Confirmed that the next refactor should preserve the existing model-loading
  path and focus on local source-like prepared-input semantics.
- Audited existing regression coverage in `tests/nodes_hy_omniweaving_test.py`.
- Added Stage 0 tests for:
  - `i2v` fallback to `reference_images` when `semantic_images` are absent
  - `t2v` propagation of `crop_start=108`, `setclip=True`, and
    `visual_input_count=0` into clip options
- Ran targeted pytest for Stage 0 text-path regression coverage and confirmed
  all selected tests passed.
- Added Stage 1 prepared-input helpers to `nodes.py`:
  - `LocalPreparedInputSpec`
  - local prompt-mode resolution
  - source-like text-path thumbnail helper
  - source-like token budget helper
  - source-like spec template builder
- Added Stage 1 tests that lock:
  - `t2v` text-only prepared spec defaults
  - `i2v` semantic-image preference and 560 thumbnail behavior
  - `i2v` fallback to text-only when visuals are absent
  - `tiv2v` image/video ordering and fallback to `editing`
- Ran targeted pytest for the new Stage 1 spec tests and related regressions.
- Migrated `_encode_prompt_components()` to use prepared-input spec ownership on
  the main non-video path:
  - `t2v`
  - `i2v`
  - `reference2v`
  - `interpolation`
- Kept legacy template-first behavior only for image-embed fallback cases where
  no text-side visual images are available yet.
- Switched `_resolve_visual_payload()` to source-like text-path thumbnail
  handling for actual text-side images.
- Updated think-mode and `reference2v` regression expectations where
  source-like text-only fallback now changes the effective prompt template or
  crop.
- Ran targeted pytest for Stage 2 and confirmed the selected regression set
  passed.
- Added `prepared_meta` plumbing from `_encode_with_parity_options()` into
  `set_clip_options()`.
- Updated the runtime patch so `_resolve_crop_start()` prefers prepared metadata
  and reports `crop_start_source=prepared_meta` when available.
- Updated the runtime patch so text-only fallback can report
  `setclip_start_source=prepared_text_only` instead of pretending visual-token
  discovery happened.
- Added regression tests for:
  - prepared metadata storage and reset
  - prepared-metadata crop-source precedence
  - prepared text-only setclip source
- Ran targeted pytest for Stage 3 and confirmed the selected regression set
  passed.
- Narrowed `_finalize_encoded_components()` so `t2v` no longer expands
  `attention_mask` for `clip_vision_output`, matching the fact that
  `HYOmniWeavingConditioning` ignores `clip_vision_output` for `t2v`.
- Narrowed `txt_in` late mask alignment so it only trims when the oversized
  prefix is an intentional all-ones expansion prefix, instead of trimming any
  longer mask indiscriminately.
- Added Stage 4 tests for:
  - no clip-vision prefix expansion on `t2v`
  - prefix-only `txt_in` trimming
  - leaving non-prefix mask mismatches untouched
- Ran targeted pytest for Stage 4 and confirmed the selected regression set
  passed.
- Added runtime debug around `forward_orig txt_mask` and `txt_in` mask
  prefix/suffix statistics.
- Real `i2v` runtime logs now show:
  - `txt_mask_len == expected_post_concat_txt_len`
  - `appears_preexpanded_for_clip=True`
  - `expected_txt_in_len != txt_mask_len`
- That confirms the next structural fix should target clip-vision mask ordering
  before tackling the remaining integrated-mask and setclip gaps.
- Added runtime debug for:
  - `151653` token positions
  - image metadata positions
  - chosen setclip boundary
  - `cond_tokens / deepstack_tokens / mask_tokens`
- Latest `i2v` runtime logs now also show:
  - `setclip` is stable on the single-image main path
  - `cond_tokens == deepstack_tokens == mask_tokens` after setclip
  - the reconstructed main-path mask is still full-coverage at `forward_orig`
    entry
- That shifts the next safest target from immediate clip-vision ordering
  changes to reducing dense no-op mask transport first, then fixing mask
  ordering.

## Update Rule

When implementation starts, update this file after each meaningful step:

- change phase status
- append a dated session note
- move completed items into `Completed`
- list the next concrete action, not a vague intention
