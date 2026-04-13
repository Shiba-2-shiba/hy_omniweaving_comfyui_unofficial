# Text Encode Refactor Spec

Last updated: 2026-04-14

## Goal

Refactor `TextEncodeHunyuanVideo15Omni` toward source-like task preparation and
crop/setclip ownership while keeping the current model-loading contract:

- keep the current single fine-tuned Qwen `safetensors` model
- keep the current ByT5 side branch contract
- do not switch back to the original HF model loader
- improve `i2v` and `t2v` quality by reducing text-path approximation

Primary target order:

1. `i2v`
2. `t2v`
3. `reference2v`
4. `interpolation`
5. `editing`
6. `tiv2v`

## Problem Statement

The current custom node preserves task names, prompt modes, and crop metadata,
but it does not preserve source-like ownership of multimodal prompt assembly.

Current gaps:

- task-specific prompt construction is split across `TASK_SPECS`,
  `_build_template()`, `_resolve_visual_payload()`, and runtime patch logic
- text-side image preparation is approximate and not centrally owned
- `attention_mask` is reconstructed from the Qwen branch instead of flowing
  directly from the integrated encode path
- final masks are expanded for `clip_vision_output` and later trimmed near
  `txt_in`, which is stable but not source-like
- `t2v` still uses the same approximate custom template route, so its quality
  ceiling is also limited

Observed stable facts from the current validated `i2v` path:

- `semantic_images` are the active text-side visual source
- `clip_vision_output` is still present and expands the final mask prefix
- `attention_mask` is reconstructed from the Qwen branch
- `txt_in` alignment trims oversized masks back to active text length

## Constraints

- Must keep the current finetuned Qwen `safetensors` model as the loaded model.
- Must not require switching to an HF runtime loader for inference.
- May use local tokenizer / processor sidecar assets such as chat template or
  preprocessor config when available.
- Must preserve the current blessed `semantic_images` path for `i2v`.
- Must preserve existing deepstack and conditioning transport unless directly
  touched by the text-path refactor.
- No new dependencies unless absolutely required.

## Design Direction

### 1. Replace template-first assembly with spec-first assembly

Introduce a local prepared-input spec layer that captures the behavior source
`prepare_input()` actually owns:

- effective task mode
- effective `crop_start`
- ordered multimodal user content
- text-side 560 thumbnail behavior
- extra tokenizer-length budget
- fallback-to-text-only decisions

Recommended structure:

```python
@dataclass
class LocalPreparedInputSpec:
    task: str
    prompt_mode: int
    crop_start: int
    user_text: str
    template: str
    ordered_visuals: list
    ordered_roles: list[str]
    token_budget_extra: int
    visual_input_count: int
    add_generation_prompt: bool = True
    used_fallback_text_only: bool = False
    meta: dict = field(default_factory=dict)
```

The spec is the source of truth. Tokenization becomes a downstream consumer of
that spec instead of implicitly re-deciding task behavior.

### 2. Keep the current `clip.tokenize()` backend

Do not rewrite the entire text stack around HF generation or full processor
execution.

Instead:

- compute source-like task semantics locally
- build a source-like ordered prompt spec
- feed that spec into the existing Comfy tokenizer path
- pass prepared metadata into runtime patch logic

This keeps the single `safetensors` loading path intact while moving the task
semantics closer to the source implementation.

### 3. Make source-like behavior explicit

#### Effective task behavior

- `t2v`: text only, `prompt_mode=1`, `crop_start=108`
- `i2v`: first-frame image to text, `prompt_mode=2`, `crop_start=92`
- `reference2v`: multi-image to text, `prompt_mode=3`, `crop_start=102`
- `interpolation`: boundary images to text, `prompt_mode=4`, `crop_start=109`
- `editing`: video-to-text, `prompt_mode=5`, `crop_start=90`
- `tiv2v`: reference image + video-to-text, `prompt_mode=6`, `crop_start=104`

Fallback behavior should stay source-like:

- if required visual/video input is missing, fall back to text-only mode rather
  than forcing a malformed multimodal path

#### Text-side image handling

- images used for text-side multimodal input must be downscaled with a
  source-like thumbnail rule
- do not upscale
- cap long side at `560`
- preserve `semantic_images` as the preferred `i2v` text-side source

#### Token budget literals

The source implementation uses fixed task-aware extra lengths. These should be
named constants in the custom node rather than hidden behavior.

Expected literals:

- `token_per_image = 400`
- video-only extra: `400 * (num_frames // 2)`
- image+video extra: `400 * (num_frames // 2) + 400 + 32`

### 4. Move crop/setclip ownership closer to prepared input

Current state:

- `crop_start` is passed explicitly
- setclip slicing is rediscovered later from token pairs
- `attention_mask` is often reconstructed from the Qwen branch
- final prefix expansion and late trim happen outside the main text encode

Target state:

- prepared input decides effective task mode and crop ownership
- runtime patch consumes prepared metadata instead of guessing from templates
- crop and setclip should operate on the same logical sequence boundaries across
  `cond`, `attention_mask`, and `all_stack_text_states`

This does not necessarily remove every final prefix adjustment, because the
current architecture still transports `clip_vision_output` outside the original
source text encoder. The goal is to minimize late corrective logic and make any
remaining prefix alignment explicit and narrow.

### 5. Restore source-like clip-vision mask ordering

The current runtime evidence now shows that `clip_vision`-related text-mask
growth is happening before `txt_in`, even though the source ordering is:

1. run `txt_in(txt, ..., txt_mask)` on text tokens
2. append `txt_byt5` if present
3. append `clip_fea` vision states
4. build the larger `attn_mask`

Current runtime logs show:

- `forward_orig txt_mask_len == expected_post_concat_txt_len`
- `forward_orig appears_preexpanded_for_clip=True`

This means the custom node is currently carrying a post-concat-length mask too
early. The next design target is:

- keep `txt_mask` at text-token length before `txt_in`
- if clip-vision-aware mask growth is still needed, move it to the stage where
  `clip_fea` is actually concatenated

This is now the highest-priority structural parity target for `i2v`.

### 6. Resolve remaining main-path gaps

After clip-vision mask ordering is corrected, the next open parity gaps remain:

- integrated `attention_mask` still missing on the main path, forcing
  `reconstructed_from_qwen_branch`
- `setclip_start=3` is still chosen heuristically from token inspection

Those should be treated as the next two source-parity targets after clip-vision
mask ordering is corrected, because current diagnostics show they are still part
of the main `i2v` path.

## Proposed Implementation Shape

### Nodes layer

Target file:

- `nodes.py`

Main changes:

- add `LocalPreparedInputSpec`
- add `_prepare_input_local_spec(...)`
- add source-like thumbnail helper for text path
- replace `_build_template()` / `_tokenize_with_template()` ownership with
  spec-first helpers
- preserve existing clip loading and `clip.tokenize()` backend
- preserve the current `semantic_images` preference for `i2v`

### Runtime patch layer

Target file:

- `runtime_patches.py`

Main changes:

- allow `set_clip_options()` to store prepared input metadata
- prefer prepared metadata over template heuristics when resolving crop/setclip
- keep `_find_setclip_start()` only as a token-level locator, not as the owner
  of task semantics
- reduce late mask correction where the spec can define ownership earlier

### Optional schema extension

Needed for source-like `editing` and `tiv2v` parity:

- add a text-side `video_frames` input to `HYOmniWeavingTextEncode`

This should be staged after `t2v` / `i2v` / `reference2v` / `interpolation`
are migrated, because those four can move first without introducing video-frame
sampling changes.

## Non-Goals

- Switching to the original HF model loader.
- Replacing the current single finetuned `safetensors` model.
- Rewriting the entire deepstack transport.
- Reworking conditioning or latent packing as the first step.
- Solving think-mode branch quality as part of this initial refactor.

## Acceptance Criteria

### Phase 1 acceptance

- `t2v`, `i2v`, `reference2v`, and `interpolation` use prepared-input spec
  ownership instead of direct custom-template ownership
- `i2v` still prefers `semantic_images`
- current validated workflows still execute

### Phase 2 acceptance

- runtime patch logic consumes prepared metadata for crop/setclip
- token-length ownership is consistent across `cond`, `attention_mask`, and
  `all_stack_text_states`
- no regression to the earlier text-mask mismatch crash class

### Phase 3 acceptance

- `editing` and `tiv2v` gain a source-like text-side video path
- video-conditioned text assembly is explicit instead of implicit

### Phase 4 acceptance

- `forward_orig txt_mask_len == expected_txt_in_len` on the main `i2v` path
- `appears_preexpanded_for_clip=False` on the main `i2v` path
- no sampler regression after moving clip-vision mask growth later

### Phase 5 acceptance

- the main `i2v` path no longer needs `attention_mask_reason=reconstructed_from_qwen_branch`
  or is demonstrably reduced to a narrower fallback-only case
- `setclip` behavior is driven by prepared semantics first and token heuristics
  second

## Verification Strategy

- add regression tests before replacing current helpers
- compare `i2v` / `t2v` token-path metadata before and after each stage
- preserve current stable workflow behavior while shrinking approximation
- keep debug logs for:
  - effective prompt mode
  - crop ownership
  - setclip ownership
  - attention mask state
  - deepstack token length
  - `forward_orig` mask dtype / branch selection
  - `expected_txt_in_len` vs `expected_post_concat_txt_len`
  - `txt_in` mask prefix and suffix statistics

## Risks

- changing task ownership without enough regression coverage could break the
  currently stable validated workflows
- `clip_vision_output` prefix handling may still need a minimal final alignment
  layer even after source-like prepared input is introduced
- changing clip-vision mask ordering without enough runtime evidence could
  silently degrade `i2v` quality even if sampling still completes
- `editing` and `tiv2v` require an explicit video-frames contract that does not
  exist yet in the current TextEncode node schema

## Decision Summary

The refactor should not chase better quality through more custom template
tuning. It should centralize source-like multimodal task preparation into a
local spec layer, keep the current single `safetensors` model path, move
crop/setclip ownership closer to that prepared-input spec, then fix the
clip-vision mask ordering so `txt_in` sees text-length masks again before
solving the remaining integrated-mask and setclip heuristic gaps.
