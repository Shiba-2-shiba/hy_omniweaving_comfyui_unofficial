# Prompt template parity notes

This note tracks how the custom node maps OmniWeaving task names onto the
original repository's `prompt_mode` definitions in
`OmniWeaving/hyvideo/models/text_encoders/__init__.py`.

## Mapping

| Task | Original prompt_mode | Original crop_start | Current custom node status |
|---|---:|---:|---|
| `t2v` | 1 | 108 | aligned |
| `i2v` | 2 | 92 | aligned |
| `reference2v` | 3 | 102 | aligned |
| `interpolation` | 4 | 109 | aligned |
| `editing` | 5 | 90 | aligned |
| `tiv2v` | 6 | 104 | aligned |

## Important detail

The biggest wording mismatch found during analysis was the `i2v` system prompt.

- **Original** starts with:
  - `Describe the key features of the input image...`
- **Current refactor target** now uses the same wording.

This matters because the project goal is not generic HunyuanVideo prompting,
but OmniWeaving-style multimodal semantics on top of ComfyUI-native inference.

## Current limitation

The custom node records the original `prompt_mode` and `crop_start` metadata,
but ComfyUI tokenization still flows through its own text-encoder path rather
than the original `prepare_input()` implementation.

So parity here currently means:

- same task-to-template intent
- same system-prompt wording
- preserved metadata for future stricter alignment work

It does **not yet** mean byte-for-byte tokenization parity with the original
processor stack.
