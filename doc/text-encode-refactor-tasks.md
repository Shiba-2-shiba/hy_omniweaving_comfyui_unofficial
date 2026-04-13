# Text Encode Refactor Tasks

Last updated: 2026-04-14

## Scope

This task list covers the TextEncode refactor defined in
`text-encode-refactor-spec.md`.

Primary objective:

- move `TextEncodeHunyuanVideo15Omni` from template-first ownership to
  source-like prepared-input ownership while keeping the current single
  finetuned `safetensors` model path

## Execution Order

### Stage 0. Lock current behavior

- [x] Identify the current regression tests that lock `i2v` text encode,
  mask reconstruction, and final prefix expansion.
- [x] Add missing regression tests for the current `t2v` text path if coverage
  is insufficient.
- [x] Add tests that capture current `semantic_images` priority in `i2v`.
- [x] Add tests that capture current `crop_start` and `setclip` option
  propagation.

Definition of done:

- the current validated behavior is locked well enough to refactor internals
  without guessing

### Stage 1. Introduce prepared-input spec

- [x] Add `LocalPreparedInputSpec` to `nodes.py`.
- [x] Add a source-like text-path thumbnail helper capped at `560`.
- [x] Add `_prepare_input_local_spec(...)`.
- [x] Encode effective fallback rules for:
  - [x] `t2v`
  - [x] `i2v`
  - [x] `reference2v`
  - [x] `interpolation`
  - [x] `editing`
  - [x] `tiv2v`
- [x] Encode task-aware token budget literals as named constants.
- [x] Add tests for spec generation, including fallback-to-text-only behavior.

Definition of done:

- task semantics are decided in one place before tokenization

### Stage 2. Migrate non-video tasks to spec-first tokenization

- [x] Replace direct `_build_template()` ownership for `t2v`.
- [x] Replace direct `_build_template()` ownership for `i2v`.
- [x] Replace direct `_build_template()` ownership for `reference2v`.
- [x] Replace direct `_build_template()` ownership for `interpolation`.
- [x] Keep the current `clip.tokenize()` backend.
- [x] Keep the current `semantic_images` priority for `i2v`.
- [x] Preserve current think-mode behavior unless blocked by the new spec layer.
- [x] Update or remove tests that only assert the old custom-template internals.

Definition of done:

- the first four tasks use the new prepared-input ownership without regressing
  current execution

### Stage 3. Push crop/setclip ownership into prepared metadata

- [x] Extend `set_clip_options()` support to accept prepared metadata.
- [x] Store prepared metadata on the patched cond-stage model.
- [x] Make crop resolution prefer prepared metadata over heuristic template-end
  inference.
- [x] Keep `_find_setclip_start()` as a token-level locator only.
- [x] Align `cond`, `attention_mask`, and `all_stack_text_states` slicing with
  the same prepared ownership.
- [x] Reduce avoidable late mask correction.
- [x] Preserve any remaining prefix alignment only where architecture still
  requires it.
- [x] Add regression tests for prepared-meta-aware crop/setclip behavior.

Definition of done:

- crop/setclip ownership is explicit and no longer primarily heuristic

### Stage 4. Evaluate late mask expansion and trim behavior

- [x] Reassess whether `clip_vision_output` prefix expansion is still needed in
  its current form after Stage 3.
- [x] Reassess whether ByT5 prefix expansion still needs the same final path.
- [x] Narrow final `txt_in` trim logic to the smallest necessary scope.
- [x] Preserve safety against the historical text-mask mismatch crash.
- [x] Add tests for any remaining final-prefix behavior.

Definition of done:

- remaining late mask correction is intentional, minimal, and documented

### Stage 5. Fix clip-vision mask ordering

- [ ] Remove `clip_vision` mask pre-expansion before `txt_in` on the main
  `i2v` path.
- [ ] Keep `txt_mask` at text-token length before `txt_in`.
- [ ] If needed, move clip-vision-aware mask growth to the stage after
  `clip_fea` is concatenated.
- [ ] Add regression coverage for:
  - [ ] `forward_orig txt_mask_len == expected_txt_in_len`
  - [ ] `appears_preexpanded_for_clip=False`
  - [ ] no sampler regression on the validated `i2v` path
- [ ] Verify the fix on real runtime logs, not just unit tests.

Definition of done:

- main-path `i2v` no longer carries post-concat-length text masks into `txt_in`

### Stage 6. Resolve remaining `attention_mask` / `setclip` gaps

- [ ] Reduce or eliminate `attention_mask_reason=reconstructed_from_qwen_branch`
  on the main `i2v` path.
- [ ] Add diagnostics that compare integrated-mask output against reconstructed
  mask output.
- [ ] Move `setclip` ownership further toward prepared semantics.
- [ ] Reduce dependence on heuristic `setclip_start=3` token inspection.
- [ ] Add regression coverage for source-like setclip behavior.

Definition of done:

- main-path `i2v` is no longer primarily driven by runtime mask reconstruction
  and heuristic setclip discovery

### Stage 7. Add explicit video-frames support

- [ ] Decide the text-side contract for `video_frames`.
- [ ] Add `video_frames` input to `HYOmniWeavingTextEncode` if needed.
- [ ] Add or plan a helper node for text-side sampled frames if direct schema
  input is too noisy.
- [ ] Implement `editing` prepared-input support.
- [ ] Implement `tiv2v` prepared-input support.
- [ ] Add regression coverage for video-conditioned text assembly.

Definition of done:

- `editing` and `tiv2v` have an explicit source-like text-side route

## Priority Notes

Highest priority:

- Stage 0
- Stage 1
- Stage 2 for `i2v` and `t2v`
- Stage 3

Important but later:

- Stage 2 for `reference2v` and `interpolation`
- Stage 4
- Stage 7

## Task Ownership Hints

### `nodes.py`

Own here:

- task semantics
- prepared-input spec creation
- source-like visual ordering
- source-like thumbnail behavior
- token budget constants

Avoid here:

- re-implementing the entire HF runtime stack

### `runtime_patches.py`

Own here:

- prepared metadata transport
- crop/setclip resolution
- encode-time slicing alignment
- minimal unavoidable final mask alignment

Avoid here:

- guessing task semantics that should have been decided earlier

## Exit Criteria

The refactor is ready for completion review when:

- `i2v` and `t2v` both run through prepared-input ownership
- crop/setclip ownership is spec-driven rather than mostly heuristic
- `txt_in` receives text-length masks again on the main `i2v` path
- `attention_mask` and `setclip` behavior are not primarily patch-reconstructed
  heuristics on the main `i2v` path
- current stable workflows still execute
- regression tests cover the replaced internal behavior
- remaining approximation is narrow, documented, and intentional
