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

These are already strengths of ComfyUI and should remain there unless parity absolutely requires otherwise:

- sampler / scheduler selection
- CFG handling
- step count / sigma schedule UI
- model loading / offloading / dtype / patcher behavior
- latent allocation and standard HunyuanVideo15 inference execution
- standard `concat_latent_image` / `concat_mask` / `clip_vision_output` conditioning transport

---

## Boundary: what should stay custom to OmniWeaving

- OmniWeaving task-specific prompt templates for:
  - `t2v`
  - `i2v`
  - `interpolation`
  - `reference2v`
  - `editing`
  - `tiv2v`
- OmniWeaving `think` rewrite path
- OmniWeaving `deepstack` / `setclip`
- OmniWeaving-specific Qwen checkpoint normalization
- OmniWeaving-specific split-QKV checkpoint normalization
- OmniWeaving 3D VAE compatibility
- task-specific latent/mask packing for `i2v` / `reference2v` / `interpolation` / `editing` / `tiv2v`

---

## High-level status

| Area | Original OmniWeaving | Current custom node | Status | Refactor direction |
|---|---|---|---|---|
| Sampler / scheduler / CFG | Pipeline-owned | ComfyUI-owned | Good | Keep in ComfyUI |
| Text encoder weights | Qwen2.5-VL + ByT5 | Dedicated dual loader exists | Good | Keep custom loader |
| Task prompt routing | Pipeline `prepare_input(prompt_mode=1..8)` | Custom task templates in `TextEncodeHunyuanVideo15Omni` | Partial | Keep custom, align more strictly |
| Think rewrite | Pipeline AR rewrite for `t2v/i2v/interpolation` | Custom `think` rewrite exists | Partial | Keep custom, verify prompt parity |
| DeepStack text states | Native in original text encoder + transformer | Recreated by runtime patching | Partial / fragile | Move toward explicit extension hook or narrower patch surface |
| ByT5 conditioning | Native in original | Uses ComfyUI HunyuanImage path + dual load | Good | Keep on ComfyUI transport path |
| Vision semantic path | Original pipeline prepares semantic images + vision states together | Workflow-supplied `clip_vision_output` only | Gap | Add OmniWeaving-aligned vision-prep node/path |
| i2v latent packing | Original task-specific | Custom node implements | Partial | Align exact mask/latent semantics |
| t2v routing | Original has task-aware text path even with no image | Custom task string exists, but no dedicated t2v conditioning node beyond empty latent path | Partial | Tighten t2v routing and invariants |
| VAE | Original `AutoencoderKLConv3D` | Local equivalent exists | Good | Keep custom VAE path |
| Transformer model detection | Original config includes Omni modules | Custom loader + runtime detection patch | Partial | Reduce patching if possible |

---

## i2v diff table

| Topic | Original flow | Current custom-node flow | Gap | Fix direction |
|---|---|---|---|---|
| Text prompt template | Qwen-VL multimodal prompt mode 2 | Custom hardcoded `i2v` template | Small/medium | Compare token boundaries and align template exactly enough for stable crop behavior |
| Visual prompt ingestion | Original text encoder consumes image objects via processor | Current node consumes `clip_vision_output.mm_projected` | Medium | Accept ComfyUI-native vision input, but add a canonical OmniWeaving preparation path so workflows stop improvising |
| Vision image preprocessing | Original pipeline resizes/crops reference image to task bucket before semantic encoding | Current repo assumes upstream `clip_vision_output` is already correct | Large | Add a custom **OmniWeaving vision-prep node** that performs bucket-aware Lanczos + center crop before CLIP/Vision encoding |
| Latent conditioning | Original VAE-encodes first frame and builds task mask | Current `HunyuanVideo15OmniConditioning` does same shape idea | Small | Verify frame count and mask polarity against original |
| Semantic image / latent roundtrip | Original derives semantic image from VAE latent roundtrip in i2v path | Current path does not reproduce this explicitly | Medium | Decide whether exact roundtrip matters for quality; if yes, add optional parity mode node |
| clip vision feature source | Original uses pipeline vision encoder output tied to prepared reference image | Current workflow can feed any `clip_vision_output` | Large | Create a narrow blessed path: `reference image -> OmniWeaving prep -> clip vision encode -> text encode + conditioning` |
| Negative conditioning | Original builds uncond input with same task mode | Current text encode node uses same task template for prompt/negative through ComfyUI conditioning path | Small | Verify negative path in real graph |

---

## t2v diff table

| Topic | Original flow | Current custom-node flow | Gap | Fix direction |
|---|---|---|---|---|
| Text prompt template | Prompt mode 1 with original system prompt | Custom `t2v` template exists | Small | Verify exact template/crop behavior |
| Think rewrite | Original supports `t2v` think rewrite | Current node supports `t2v` think rewrite | Small | Validate output length / stopping behavior |
| Visual inputs absent behavior | Original still routes through task-specific text path and zero-vision behavior | Current node can run without `clip_vision_output`; conditioning node returns latent only | Medium | Ensure no accidental image-path assumptions leak into t2v |
| t2v conditioning | Original uses t2v multitask mask, no cond latents | Current conditioning node returns zero latent and optionally forwards `clip_vision_output` | Good | Keep custom node thin |
| Resolution/task routing | Original resolves bucket/aspect handling in pipeline | Current graph leaves width/height selection to workflow | Medium | Add docs or helper node for OmniWeaving-recommended t2v resolution selection, but keep actual sampling in ComfyUI |

---

## runtime_patches diff table

| Patch area | Why it exists now | Is it logically custom-node responsibility? | Keep / replace |
|---|---|---|---|
| `_patch_hunyuan_image_te` | Added `deepstack` + `setclip` support to ComfyUI Hunyuan image text encoder globally | Partly yes, but monkey-patching whole encode path is fragile | **Removed**: text-encoder patching now happens per loaded clip instance |
| `_patch_qwen25_think_generation` | Supplies safe stop tokens for think generation | Mostly custom behavior | Keep for now, but isolate as the smallest possible patch |
| `_patch_model_detection` | Marked HunyuanVideo model config with `deepstack` based on state dict | Compatibility glue | **Removed**: loader now attaches `mm_in` directly after model load |
| `_patch_model_base` | Added `all_stack_text_states` to extra conds globally | Necessary for deepstack transport | **Removed**: loader now patches `extra_conds` per loaded model instance |
| `_patch_hunyuan_video_model` | Applied early-block text-state injection globally | Core OmniWeaving model behavior | **Removed**: deepstack early-block injection now rides on a per-model DIFFUSION_MODEL wrapper |
| `_patch_autoencoder_legacy` | Supports decoder ddconfig path for VAE loading | Mostly compatibility shim | Can remain until VAE path is fully custom-owned |

---

## Recommended refactor lanes

### Lane 1 — stabilize the i2v blessed path

Create one canonical path and treat all other paths as unsupported for parity:

1. `reference image`
2. `OmniWeaving image prep`  
   - bucket-aware target size
   - Lanczos resize
   - center crop
3. `CLIP/Vision encode`
4. `HY OmniWeaving Text Encode`
5. `HY OmniWeaving Conditioning`
6. stock ComfyUI sampler / scheduler / CFG

This is the main fix for the current “visual condition path drift” risk.

### Lane 2 — tighten t2v routing

- explicitly document / enforce the `t2v` no-image path
- keep sampler/scheduler entirely stock
- verify template/crop behavior against original prompt mode 1

### Lane 3 — shrink runtime patch surface

Priority order:

1. remove detection patch if loader can inject `deepstack` config explicitly
2. replace `model_base` patch with a smaller wrapper-compatible cond injection path
3. isolate transformer deepstack patch behind one clearly named adapter layer
4. leave think-generation stop-token patch until a cleaner core hook exists

---

## Concrete missing pieces to implement next

1. **Diff artifact for prompt/template parity**
   - status: implemented in `prompt_template_parity.md`
   - remaining work:
     - decide how much stricter token-level parity should become versus keeping ComfyUI-native tokenization ergonomics

2. **OmniWeaving image preparation helper/node**
   - status: implemented as `HY OmniWeaving Image Prep`
   - remaining work:
     - wire into example workflows
     - ensure docs consistently treat this as the blessed i2v path

3. **Exact i2v mask/latent parity verification**
   - confirm mask polarity
   - confirm temporal fill behavior
   - confirm latent frame placement

4. **DeepStack boundary cleanup design**
   - identify whether `all_stack_text_states` can be injected via existing ComfyUI wrappers instead of patching `_forward`

---

## Working conclusion

The repo is already on the right architectural direction **if the goal is “OmniWeaving semantics on top of ComfyUI-native inference.”**

The main remaining problems are not sampler/scheduler-related; they are:

- exact **task routing parity**
- exact **i2v visual-conditioning parity**
- reducing the amount of **runtime monkey patching**

So the next refactor should focus on **conditioning-path correctness**, not on rebuilding original inference internals that ComfyUI already does well.
