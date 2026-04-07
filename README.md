# hy_omniweaving_comfyui_unofficial

Unofficial ComfyUI custom node for running HY-OmniWeaving on top of stock ComfyUI.

## Scope

- first-pass focus on `i2v`
- HY-OmniWeaving-oriented text encoding
- ByT5 / visual-input parity guards
- optional `think` prompt expansion
- `deepstack` / `setclip` option plumbing
- Omni-style conditioning with Lanczos + center-crop image preparation

## Current limitations

- full task parity beyond `i2v` is not done yet
- full resolution-bucket parity is not done yet
- some lower-level parity still relies on runtime patching into ComfyUI internals

## Install

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/hy_omniweaving_comfyui_unofficial.git
```

Then restart ComfyUI.

## Important note

This package currently applies runtime patches at import time in order to supply
HY-OmniWeaving-specific parity hooks that stock ComfyUI does not expose
directly. The long-term goal is to reduce these patch points further.
