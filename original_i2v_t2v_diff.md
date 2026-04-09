# OmniWeaving original vs ComfyUI custom-node diff

## Goal

Refactor toward:

- keeping **sampling / scheduler / CFG / step count / model loading / VRAM management** inside stock ComfyUI
- moving only **OmniWeaving-specific conditioning + text/vision semantics + minimal model hooks** into this custom node
- reducing `runtime_patches.py` dependence over time

Primary target parity:

1. `i2v`
2. `t2v`

---

## Boundary: what should stay in stock ComfyUI

These remain strengths of ComfyUI and should stay there unless parity requires
otherwise:

- sampler / scheduler selection
- CFG handling
- step count / sigma schedule UI
- model loading / offloading / dtype / patcher behavior
- latent allocation and standard HunyuanVideo15 inference execution
- standard `concat_latent_image` / `concat_mask` / `clip_vision_output`
  conditioning transport

## Boundary: what should stay custom to OmniWeaving

- OmniWeaving task-specific prompt templates
- OmniWeaving `think` rewrite path
- OmniWeaving `deepstack` / `setclip`
- OmniWeaving-specific Qwen checkpoint normalization
- OmniWeaving-specific split-QKV checkpoint normalization
- OmniWeaving 3D VAE compatibility
- task-specific latent/mask packing for `i2v` / `reference2v` /
  `interpolation` / `editing` / `tiv2v`

---

## Current high-level status

| Area | Original OmniWeaving | Current custom node | Status | Notes |
|---|---|---|---|---|
| Sampler / scheduler / CFG | Pipeline-owned | ComfyUI-owned | Good | Keep in ComfyUI |
| Text encoder weights | Qwen2.5-VL + ByT5 | Dedicated dual loader exists | Good | Keep custom loader |
| Task prompt routing | Pipeline `prepare_input(prompt_mode=1..8)` | Custom task templates in `TextEncodeHunyuanVideo15Omni` | Good enough for current `i2v` / `t2v` | Still not byte-for-byte processor parity |
| Think rewrite | Pipeline AR rewrite for `t2v/i2v/interpolation` | Custom `think` rewrite exists | Partial | Keep custom, refine only if needed |
| DeepStack text states | Native in original text encoder + transformer | Recreated by runtime patching | Structurally fixed, numerically limited | Transport works, connector weights do not |
| ByT5 conditioning | Native in original | Uses ComfyUI HunyuanImage path + dual load | Good | Keep on ComfyUI transport path |
| Vision semantic path | Original pipeline prepares semantic images + vision states together | `Image Prep` + `I2V Semantic Images` + stock CLIP-Vision path | Good for current `i2v` | This is now the blessed path |
| i2v latent packing | Original task-specific | Custom node implements | Good enough for current `i2v` | Execution issue from invalid `ref_latent` removed |
| t2v routing | Original has task-aware text path even with no image | Custom text-only path plus explicit zero-conditioning | Good | Keep path thin |
| VAE | Original `AutoencoderKLConv3D` | Local equivalent exists | Good | Keep custom VAE path |
| Transformer model detection | Original config includes Omni modules | Loader + runtime attach | Partial | Still patch-based, but much narrower than before |

---

## i2v diff table

| Topic | Original flow | Current custom-node flow | Current assessment |
|---|---|---|---|
| Text prompt template | Qwen-VL multimodal prompt mode 2 | Custom `i2v` template with original wording and `crop_start=92` metadata | Good enough for current validated path |
| Visual prompt ingestion | Original text encoder consumes image objects via processor | Current node consumes `semantic_images` for text-side multimodal input | Correct practical direction |
| Vision image preprocessing | Original pipeline resizes/crops reference image to task bucket before semantic encoding | `HY OmniWeaving Image Prep` performs Lanczos + center crop before semantic and vision paths | Correct practical direction |
| Semantic image / latent roundtrip | Original derives semantic image from VAE latent roundtrip | `HY OmniWeaving I2V Semantic Images` reproduces this path | Implemented |
| Latent conditioning | Original VAE-encodes first frame and builds task mask | `HYOmniWeavingConditioning` builds `concat_latent_image` and `concat_mask` | Implemented and currently working |
| CLIP-Vision path | Original uses vision states tied to the prepared semantic image | Current blessed workflow uses stock CLIP-Vision on `semantic_images` | Implemented and currently working |
| Extra guiding path | Original runtime combines multiple guidance mechanisms | Current path uses `guiding_frame_index` but no longer sends the invalid `ref_latent` payload | Fixed execution bug; keep this direction unless runtime contract changes |
| Remaining parity gap | Original processor stack defines final token slicing | Current path still shows a small crop/setclip token mismatch | Remaining quality/parity task |

---

## t2v diff table

| Topic | Original flow | Current custom-node flow | Current assessment |
|---|---|---|---|
| Text prompt template | Prompt mode 1 with original system prompt | Custom `t2v` template with `crop_start=108` metadata | Good enough for current validated path |
| Think rewrite | Original supports `t2v` think rewrite | Current node supports `t2v` think rewrite | Acceptable |
| Visual inputs absent behavior | Original still uses task-aware text route with no image | Current node runs with `use_visual_inputs=false` and explicit zero-conditioning | Correct practical direction |
| t2v conditioning | Original uses t2v multitask mask and no image conditioning | Current conditioning node returns zero latent and full mask | Implemented and currently working |
| Remaining parity gap | Original stack still owns exact processor/token semantics | Current path still shows a small crop/setclip token mismatch in logs | Quality/parity follow-up only |

---

## runtime_patches diff table

| Patch area | Current status | Assessment |
|---|---|---|
| text-encoder patching | per loaded clip instance | Good reduction from earlier global patching |
| extra cond injection | per loaded model instance | Good reduction from earlier global model-base patching |
| deepstack diffusion injection | per-model `DIFFUSION_MODEL` wrapper | Acceptable for now |
| think-generation patch | still patch-based | Keep until a cleaner hook exists |
| VAE compatibility patching | still partially compatibility-driven | Acceptable while custom VAE ownership matures |

---

## What changed materially in this pass

### 1. The blessed `i2v` path is now real, not just planned

The custom node now has a concrete, working path that matches the intended
shape of the original flow closely enough to generate clear output:

1. reference image
2. `HY OmniWeaving Image Prep`
3. `HY OmniWeaving I2V Semantic Images`
4. stock CLIP-Vision encode
5. `HY OmniWeaving Text Encode`
6. `HY OmniWeaving Conditioning`
7. stock ComfyUI sampler / scheduler / CFG

### 2. The `t2v` route is now explicitly thin

The custom node now treats `t2v` as:

1. task-specific text route
2. explicit zero-conditioning
3. stock ComfyUI inference

This is the intended boundary.

### 3. The old `i2v` execution blocker was integration-side

The previous `i2v` crash was not a sampler problem or a VAE problem. It was a
bad `ref_latent` payload for the active ComfyUI HunyuanVideo runtime.

Removing that invalid path was the correct short-term fix.

---

## Remaining refactor lanes

### Lane 1 — preserve the current blessed workflows

- keep `i2v` on prepared image -> semantic image -> text/vision conditioning
- keep `t2v` text-only
- do not casually reintroduce invalid extra conditions

### Lane 2 — tighten crop/setclip parity

- investigate the remaining small `cond_tokens` vs `deepstack_tokens` mismatch
- compare current slicing rules against original `prepare_input()` behavior

### Lane 3 — wait for corrected deepstack weights

- `mm_in.linear_2` remains all-zero in the tested public checkpoints
- treat any future model-provider fix as a new evaluation point before making
  larger architectural changes

---

## Working conclusion

The repo is now on the right architectural direction **and** is currently
usable for practical `i2v` / `t2v` work.

The main remaining problems are no longer "can it run?" problems. They are:

- exact crop/setclip parity
- motion-heavy quality refinement
- restored deepstack effectiveness once corrected weights exist

So the next refactor should focus on **parity refinement**, not on rebuilding
inference internals that ComfyUI already handles well.
