# Parity Notes

This note tracks how the custom node maps onto the original OmniWeaving
repository and where parity is still approximate rather than exact.

## Goal

Refactor toward:

- keeping **sampling / scheduler / CFG / step count / model loading / VRAM
  management** inside stock ComfyUI
- moving only **OmniWeaving-specific conditioning + text/vision semantics +
  minimal model hooks** into this custom node

Primary target parity:

1. `i2v`
2. `t2v`

## Prompt template mapping

The custom node preserves the original task-to-`prompt_mode` mapping and
tracked `crop_start` values.

| Task | Original prompt_mode | Original crop_start | Current custom node status |
|---|---:|---:|---|
| `t2v` | 1 | 108 | aligned in current validated workflow |
| `i2v` | 2 | 92 | aligned in current validated workflow |
| `reference2v` | 3 | 102 | mapped |
| `interpolation` | 4 | 109 | mapped |
| `editing` | 5 | 90 | mapped |
| `tiv2v` | 6 | 104 | mapped |

Current confirmed behavior:

- the `i2v` system prompt matches the original wording closely enough to avoid
  the earlier semantic drift from prompt mismatch
- in the current validated `i2v` path, text-side multimodal input is driven by
  `semantic_images`, not `clip_vision_output.mm_projected`
- in the current validated `t2v` path, `use_visual_inputs=false` and the node
  follows the intended text-only route

## High-level diff

| Area | Original OmniWeaving | Current custom node | Status | Notes |
|---|---|---|---|---|
| Sampler / scheduler / CFG | Pipeline-owned | ComfyUI-owned | Good | Keep in ComfyUI |
| Text encoder weights | Qwen2.5-VL + ByT5 | Dedicated dual loader exists | Good | Keep custom loader |
| Vision encoder / projector | SigLIP + Redux image embedder | Dedicated Redux vision node exists | Good | Keep custom loader-style path |
| Task prompt routing | Pipeline `prepare_input(prompt_mode=1..8)` | Custom task templates in `TextEncodeHunyuanVideo15Omni` | Good enough for current `i2v` / `t2v` | Not byte-for-byte processor parity |
| Think rewrite | Pipeline AR rewrite for `t2v/i2v/interpolation` | Custom `think` rewrite exists | Partial | Keep custom, refine only if needed |
| DeepStack text states | Native in original text encoder + transformer | Recreated by runtime patching | Structurally fixed, numerically active in tested run | Current debug logs show non-zero projection |
| ByT5 conditioning | Native in original | Uses ComfyUI HunyuanImage path + dual load | Good | Transport path preserved; real ByT5-active run still needs validation |
| Vision semantic path | Original prepares semantic images + vision states together | `Image Prep` + `I2V Semantic Images` + `Redux Vision Encode` | Good for current `i2v` | This is the blessed path |
| Text mask handling | Original stack keeps prompt and mask logic together | Runtime patch reconstructs Qwen text masks, expands final prefixes, and aligns `txt_in` masks | Acceptable for now | Stable in validated `i2v`, not processor-identical |
| i2v latent packing | Original task-specific | Custom node implements | Good enough for current `i2v` | Invalid `ref_latent` path removed |
| t2v routing | Original has task-aware text path with no image | Custom text-only path plus explicit zero-conditioning | Good | Keep path thin |
| VAE | Original `AutoencoderKLConv3D` | Local equivalent exists | Good | Keep custom VAE path |
| Transformer model detection | Original config includes Omni modules | Loader + runtime attach | Partial | Still patch-based, but much narrower than before |

## i2v flow

| Topic | Original flow | Current custom-node flow | Current assessment |
|---|---|---|---|
| Text prompt template | Qwen-VL multimodal prompt mode 2 | Custom `i2v` template with original wording and `crop_start=92` metadata | Good enough for current validated path |
| Visual prompt ingestion | Original text encoder consumes image objects via processor | Current node consumes `semantic_images` for text-side multimodal input | Correct practical direction |
| Vision image preprocessing | Original pipeline resizes/crops reference image before semantic encoding | `HY OmniWeaving Image Prep` performs Lanczos + center crop before semantic and vision paths | Correct practical direction |
| Vision encoder path | Original uses SigLIP + Redux image embedder | `HY OmniWeaving Redux Vision Encode` loads selected encoder/embedder checkpoints from `clip_vision` | Implemented |
| Semantic image / latent roundtrip | Original derives semantic image from VAE roundtrip | `HY OmniWeaving I2V Semantic Images` reproduces this path | Implemented |
| Latent conditioning | Original VAE-encodes first frame and builds task mask | `HY OmniWeaving Conditioning` builds `concat_latent_image` and `concat_mask` | Implemented and working |
| Vision output transport | Original uses vision states tied to prepared semantic image | Current workflow uses Redux-backed `clip_vision_output` while text-side image input still prefers `semantic_images` | Implemented and working |
| Text-mask transport | Original keeps mask internally through prepare / encode | Current path reconstructs Qwen text mask, expands final prefixes, and trims early `txt_in` masks | Implemented and working in validated run |
| Remaining parity gap | Original processor stack defines exact token slicing | Current path still uses custom reconstruction / expansion logic | Remaining parity task |

## t2v flow

| Topic | Original flow | Current custom-node flow | Current assessment |
|---|---|---|---|
| Text prompt template | Prompt mode 1 with original system prompt | Custom `t2v` template with `crop_start=108` metadata | Good enough for current validated path |
| Think rewrite | Original supports `t2v` think rewrite | Current node supports `t2v` think rewrite | Acceptable |
| Visual inputs absent behavior | Original still uses task-aware text route with no image | Current node runs with `use_visual_inputs=false` and explicit zero-conditioning | Correct practical direction |
| t2v conditioning | Original uses t2v multitask mask and no image conditioning | Current conditioning node returns zero latent and full mask | Implemented and working |
| Remaining parity gap | Original stack still owns exact processor/token semantics | Current path still relies on custom mask reconstruction and branch-content shaping | Quality/parity follow-up only |

## Current parity limits

Parity here currently means:

- same task-to-template intent
- same system-prompt wording
- same `crop_start` metadata
- same high-level workflow expectation about whether images should be present
- a validated custom mask path that no longer crashes in the current `i2v`
  route

It does **not** yet mean:

- byte-for-byte tokenization parity with the original processor stack
- a proven ByT5-active final mask path from a real run
- public-repo-equivalent `merge_hidden` behavior, since that path is custom

## Current follow-up items

- validate a real run with active `conditioning_byt5small`
- continue comparing crop / setclip slicing against the original
  `prepare_input()` behavior
- keep treating `merge_hidden` branch content quality as a separate issue from
  structural parity
