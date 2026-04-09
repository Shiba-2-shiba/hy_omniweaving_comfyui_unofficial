# Prompt template parity notes

This note tracks how the custom node maps OmniWeaving task names onto the
original repository's `prompt_mode` definitions in
`OmniWeaving/hyvideo/models/text_encoders/__init__.py`.

## Mapping

| Task | Original prompt_mode | Original crop_start | Current custom node status |
|---|---:|---:|---|
| `t2v` | 1 | 108 | aligned and validated in current best-known workflow |
| `i2v` | 2 | 92 | aligned and validated in current best-known workflow |
| `reference2v` | 3 | 102 | mapped |
| `interpolation` | 4 | 109 | mapped |
| `editing` | 5 | 90 | mapped |
| `tiv2v` | 6 | 104 | mapped |

## Confirmed current behavior

- The custom node preserves the original task-to-`prompt_mode` mapping and
  carries the original `crop_start` values into the patched text encoder path.
- The `i2v` system prompt now matches the original wording closely enough to
  avoid the earlier semantic drift caused by prompt mismatch.
- In the current validated `i2v` path, text-side multimodal input is driven by
  `semantic_images`, not by `clip_vision_output.mm_projected`.
- In the current validated `t2v` path, `use_visual_inputs=false` and the node
  follows the intended text-only task route.

## Important limitation

Parity here currently means:

- same task-to-template intent
- same system-prompt wording
- same `crop_start` metadata
- same high-level workflow expectation about whether images should be present

It does **not** yet mean byte-for-byte tokenization parity with the original
processor stack.

## Remaining mismatch to investigate

Current `i2v` / `t2v` debug logs still show a small token-count mismatch after
crop/setclip:

- `cond_tokens`
- `deepstack_tokens`

The mismatch is small and no longer blocks execution, but it is still a useful
parity-quality signal and should be treated as a remaining follow-up item.
