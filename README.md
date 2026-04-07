# hy-omniweaving-comfyui

Custom-node extraction lane for HY-OmniWeaving support on top of ComfyUI.

Current scope:

- `i2v` first
- HY-OmniWeaving-oriented text encoding
- ByT5/visual-input parity guards
- optional `think` prompt expansion
- `deepstack` / `setclip` option plumbing
- Omni-style conditioning with Lanczos + center-crop image preparation

Current non-goals:

- full task parity beyond `i2v`
- full resolution-bucket parity
- UI/workflow backward compatibility

Current status:

- This package is the new refactor target for HY-OmniWeaving-specific logic.
- Some lower-level parity support still lives in the current ComfyUI core checkout while the extraction is in progress.
- Future work should continue here rather than adding more Omni-specific behavior to generic ComfyUI Hunyuan modules.
- `runtime_patches.py` is the extraction hook for eventually moving the remaining core edits behind custom-node-owned monkey patches.
