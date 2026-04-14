"""
Runtime patch entrypoint for HY-OmniWeaving support.

Current state:
- Some parity support still exists in the ComfyUI core checkout.
- This module progressively absorbs the easier pieces behind custom-node-owned
  monkey patches so that future refactors have a single home.
- Deepstack support is now attached per loaded model/clip instance where
  possible; remaining global patches are limited to think-generation stop
  tokens and legacy autoencoder compatibility.
"""

from __future__ import annotations
import numbers
import types
import logging
import math

import torch
from torch import nn

_HY_OMNIWEAVING_RUNTIME_PATCHES_READY = False


def _debug_enabled() -> bool:
    import os
    return os.getenv("HY_OMNIWEAVING_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str, *args):
    if _debug_enabled():
        logging.info("[HY-OmniWeaving:debug] " + message, *args)


def _shape_of(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    return None

def _mask_summary(value):
    if not torch.is_tensor(value):
        return {
            "shape": None,
            "dtype": getattr(value, "dtype", None),
            "nonzero": None,
            "all_ones": None,
            "min": None,
            "max": None,
        }
    value_float = value.float()
    return {
        "shape": tuple(value.shape),
        "dtype": value.dtype,
        "nonzero": int(torch.count_nonzero(value).item()),
        "all_ones": bool(torch.all(value == 1).item()),
        "min": float(value_float.min().item()),
        "max": float(value_float.max().item()),
    }


def _norm_of(value):
    if torch.is_tensor(value):
        return float(value.float().norm().item())
    return None


def _rounded_temporal_list(value, digits: int = 4):
    if value is None:
        return None
    return [round(float(item), digits) for item in value]


def _active_temporal_indices(values, threshold: float = 0.5, invert: bool = False):
    if values is None:
        return None
    out = []
    for index, item in enumerate(values):
        active = float(item) > threshold
        if invert:
            active = not active
        if active:
            out.append(index)
    return out


def _temporal_mask_vector(mask):
    if not torch.is_tensor(mask) or mask.ndim < 5:
        return None
    return mask[0, 0, :, 0, 0].detach().float().cpu().tolist()


def _temporal_latent_energy(latent):
    if not torch.is_tensor(latent) or latent.ndim < 5:
        return None
    return latent[0].detach().float().abs().sum(dim=(0, 2, 3)).cpu().tolist()


def _split_concat_tensor(c_concat):
    if not torch.is_tensor(c_concat) or c_concat.ndim < 5 or c_concat.shape[1] < 1:
        return None, None
    if c_concat.shape[1] == 1:
        return None, c_concat
    return c_concat[:, :-1], c_concat[:, -1:]


def _callable_debug_name(value):
    if value is None:
        return "None"
    func = getattr(value, "__func__", value)
    module = getattr(func, "__module__", None) or "<unknown>"
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", type(func).__name__))
    bound_self = getattr(value, "__self__", None)
    if bound_self is not None:
        return f"{module}.{qualname} bound_to={type(bound_self).__name__}"
    return f"{module}.{qualname}"


def _is_effectively_zero(value):
    return torch.is_tensor(value) and bool(torch.count_nonzero(value).item() == 0)


class _CONDDeepstackTextStates:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, **kwargs):
        if self.cond.shape[1] == batch_size:
            return self._copy_with(self.cond)
        current = self.cond.shape[1]
        if current > batch_size:
            return self._copy_with(self.cond[:, :batch_size])
        repeat_factor = math.ceil(batch_size / current)
        repeated = self.cond.repeat(1, repeat_factor, 1, 1)[:, :batch_size]
        return self._copy_with(repeated)

    def can_concat(self, other):
        if self.cond.dim() != other.cond.dim():
            return False
        if self.cond.device != other.cond.device:
            logging.warning("WARNING: deepstack conds not on same device, skipping concat.")
            return False
        if self.cond.shape[0] != other.cond.shape[0]:
            return False
        if self.cond.shape[2:] != other.cond.shape[2:]:
            return False
        return True

    def concat(self, others):
        conds = [self.cond] + [x.cond for x in others]
        return torch.cat(conds, dim=1)

    def size(self):
        hidden = self.cond.shape[-1] if self.cond.dim() > 0 else 1
        return [self.cond.shape[1], hidden, math.prod(self.cond.shape) // max(1, hidden)]


class _TextProjection(nn.Module):
    def __init__(self, in_channels, hidden_size, linear_cls=nn.Linear, dtype=None, device=None):
        super().__init__()
        self.linear_1 = linear_cls(in_channels, hidden_size, bias=True, device=device, dtype=dtype)
        self.act_1 = nn.SiLU()
        self.linear_2 = linear_cls(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def extract_hy_omniweaving_mm_in_state_dict(sd: dict) -> dict:
    return {
        key[len("mm_in."):]: value
        for key, value in sd.items()
        if key.startswith("mm_in.")
    }


def _ensure_hy_omniweaving_extra_conds_support(model):
    if getattr(model, "_hy_omniweaving_extra_conds_patched", False):
        return False

    original_extra_conds = model.extra_conds
    original_concat_cond = getattr(model, "concat_cond", None)
    original_apply_model = getattr(model, "_apply_model", None)

    def patched_extra_conds(self, **kwargs):
        concat_latent_image = kwargs.get("concat_latent_image", None)
        concat_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        guiding_frame_index = kwargs.get("guiding_frame_index", None)
        concat_mask_vector = _temporal_mask_vector(concat_mask)
        expected_model_mask_vector = None
        if concat_mask_vector is not None:
            expected_model_mask_vector = [1.0 - item for item in concat_mask_vector]
        out = original_extra_conds(**kwargs)
        all_stack_text_states = kwargs.get("all_stack_text_states", None)
        if all_stack_text_states is not None:
            out["all_stack_text_states"] = _CONDDeepstackTextStates(all_stack_text_states)
        c_concat = out.get("c_concat", None)
        c_concat_value = getattr(c_concat, "cond", None)
        if any(value is not None for value in (concat_latent_image, concat_mask, guiding_frame_index, all_stack_text_states)):
            _debug_log(
                "extra_conds payload concat_latent_shape=%s concat_latent_active_frames=%s concat_mask_shape=%s concat_mask=%s concat_mask_zero_frames=%s expected_model_mask=%s expected_model_mask_active=%s guiding_frame_index=%s all_stack_text_states_shape=%s out_keys=%s c_concat_shape=%s",
                _shape_of(concat_latent_image),
                _active_temporal_indices(_temporal_latent_energy(concat_latent_image), threshold=1e-6),
                _shape_of(concat_mask),
                _rounded_temporal_list(concat_mask_vector),
                _active_temporal_indices(concat_mask_vector, invert=True),
                _rounded_temporal_list(expected_model_mask_vector),
                _active_temporal_indices(expected_model_mask_vector),
                guiding_frame_index,
                _shape_of(all_stack_text_states),
                sorted(out.keys()),
                _shape_of(c_concat_value),
            )
        return out

    def patched_concat_cond(self, **kwargs):
        c_concat = original_concat_cond(**kwargs)
        image_part, mask_part = _split_concat_tensor(c_concat)
        if torch.is_tensor(c_concat):
            _debug_log(
                "concat_cond output c_concat_shape=%s c_concat_norm=%s image_part_shape=%s image_frame_energy=%s image_active_frames=%s mask_part_shape=%s mask=%s mask_active_frames=%s",
                _shape_of(c_concat),
                _norm_of(c_concat),
                _shape_of(image_part),
                _rounded_temporal_list(_temporal_latent_energy(image_part)),
                _active_temporal_indices(_temporal_latent_energy(image_part), threshold=1e-6),
                _shape_of(mask_part),
                _rounded_temporal_list(_temporal_mask_vector(mask_part)),
                _active_temporal_indices(_temporal_mask_vector(mask_part)),
            )
        return c_concat

    def patched_apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        image_part, mask_part = _split_concat_tensor(c_concat)
        diffusion_model = getattr(self, "diffusion_model", None)
        if diffusion_model is not None:
            diffusion_model._hy_last_apply_model_c_concat_channels = int(c_concat.shape[1]) if torch.is_tensor(c_concat) and c_concat.ndim >= 2 else 0
        if torch.is_tensor(c_concat):
            _debug_log(
                "apply_model inputs noise_shape=%s noise_norm=%s c_concat_shape=%s c_concat_norm=%s c_concat_image_shape=%s c_concat_image_frame_energy=%s c_concat_image_active_frames=%s c_concat_mask_shape=%s c_concat_mask=%s c_concat_mask_active_frames=%s",
                _shape_of(x),
                _norm_of(x),
                _shape_of(c_concat),
                _norm_of(c_concat),
                _shape_of(image_part),
                _rounded_temporal_list(_temporal_latent_energy(image_part)),
                _active_temporal_indices(_temporal_latent_energy(image_part), threshold=1e-6),
                _shape_of(mask_part),
                _rounded_temporal_list(_temporal_mask_vector(mask_part)),
                _active_temporal_indices(_temporal_mask_vector(mask_part)),
            )
        return original_apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    model.extra_conds = types.MethodType(patched_extra_conds, model)
    if original_concat_cond is not None:
        model.concat_cond = types.MethodType(patched_concat_cond, model)
    if original_apply_model is not None:
        model._apply_model = types.MethodType(patched_apply_model, model)
    model._hy_omniweaving_extra_conds_patched = True
    logging.info("HY-OmniWeaving attached instance-local extra_conds support.")
    return True


def _ensure_hy_omniweaving_txt_mask_alignment_support(diffusion_model):
    txt_in = getattr(diffusion_model, "txt_in", None)
    if txt_in is None:
        return False
    if getattr(txt_in, "_hy_omniweaving_mask_alignment_patched", False):
        return False

    original_forward = txt_in.forward

    def _mask_prefix_debug_stats(mask, target_length: int):
        if not torch.is_tensor(mask) or mask.ndim < 2:
            return {
                "extra_tokens": 0,
                "prefix_shape": None,
                "prefix_non_one_count": None,
                "prefix_min": None,
                "prefix_max": None,
                "suffix_shape": None,
                "suffix_nonzero_count": None,
                "suffix_min": None,
                "suffix_max": None,
            }
        extra_tokens = max(0, int(mask.shape[-1] - target_length))
        if extra_tokens <= 0:
            suffix = mask
            suffix_float = suffix.float()
            return {
                "extra_tokens": 0,
                "prefix_shape": None,
                "prefix_non_one_count": 0,
                "prefix_min": None,
                "prefix_max": None,
                "suffix_shape": _shape_of(suffix),
                "suffix_nonzero_count": int(torch.count_nonzero(suffix).item()),
                "suffix_min": float(suffix_float.min().item()),
                "suffix_max": float(suffix_float.max().item()),
            }
        prefix = mask[..., :extra_tokens]
        suffix = mask[..., -target_length:]
        prefix_float = prefix.float()
        suffix_float = suffix.float()
        prefix_non_one_count = int(torch.count_nonzero(prefix != 1).item())
        return {
            "extra_tokens": extra_tokens,
            "prefix_shape": _shape_of(prefix),
            "prefix_non_one_count": prefix_non_one_count,
            "prefix_min": float(prefix_float.min().item()),
            "prefix_max": float(prefix_float.max().item()),
            "suffix_shape": _shape_of(suffix),
            "suffix_nonzero_count": int(torch.count_nonzero(suffix).item()),
            "suffix_min": float(suffix_float.min().item()),
            "suffix_max": float(suffix_float.max().item()),
        }

    def patched_forward(self, x, *args, **kwargs):
        mask = None
        if len(args) >= 2:
            mask = args[1]
        elif "mask" in kwargs:
            mask = kwargs.get("mask")

        effective_mask = mask
        if (
            torch.is_tensor(mask)
            and mask.ndim >= 2
            and torch.is_tensor(x)
            and x.ndim >= 2
            and mask.shape[-1] > x.shape[1]
        ):
            prefix_stats = _mask_prefix_debug_stats(mask, x.shape[1])
            effective_mask = mask[..., -x.shape[1]:]
            _debug_log(
                "txt_in mask alignment x_shape=%s original_mask_shape=%s effective_mask_shape=%s extra_tokens=%s "
                "prefix_shape=%s prefix_non_one_count=%s prefix_min=%s prefix_max=%s "
                "suffix_shape=%s suffix_nonzero_count=%s suffix_min=%s suffix_max=%s",
                _shape_of(x),
                _shape_of(mask),
                _shape_of(effective_mask),
                prefix_stats["extra_tokens"],
                prefix_stats["prefix_shape"],
                prefix_stats["prefix_non_one_count"],
                prefix_stats["prefix_min"],
                prefix_stats["prefix_max"],
                prefix_stats["suffix_shape"],
                prefix_stats["suffix_nonzero_count"],
                prefix_stats["suffix_min"],
                prefix_stats["suffix_max"],
            )

        if len(args) >= 2:
            args = list(args)
            args[1] = effective_mask
            args = tuple(args)
        elif "mask" in kwargs:
            kwargs = kwargs.copy()
            kwargs["mask"] = effective_mask

        return original_forward(x, *args, **kwargs)

    txt_in.forward = types.MethodType(patched_forward, txt_in)
    txt_in._hy_omniweaving_mask_alignment_patched = True
    logging.info("HY-OmniWeaving attached instance-local txt_in mask alignment support.")
    return True


def _ensure_hy_omniweaving_forward_orig_txt_mask_debug_support(diffusion_model):
    forward_orig = getattr(diffusion_model, "forward_orig", None)
    if forward_orig is None:
        return False
    if getattr(diffusion_model, "_hy_omniweaving_forward_orig_txt_mask_debug_patched", False):
        return False

    def patched_forward_orig(self, *args, **kwargs):
        img = args[0] if len(args) >= 1 else kwargs.get("img")
        context = args[2] if len(args) >= 3 else kwargs.get("context")
        txt_mask = None
        if len(args) >= 5:
            txt_mask = args[4]
        elif "txt_mask" in kwargs:
            txt_mask = kwargs.get("txt_mask")
        txt_byt5 = args[7] if len(args) >= 8 else kwargs.get("txt_byt5")
        clip_fea = args[8] if len(args) >= 9 else kwargs.get("clip_fea")

        txt_mask_shape = _shape_of(txt_mask)
        txt_mask_dtype = getattr(txt_mask, "dtype", None)
        txt_mask_is_floating = bool(torch.is_floating_point(txt_mask)) if torch.is_tensor(txt_mask) else None
        txt_mask_nonzero = int(torch.count_nonzero(txt_mask).item()) if torch.is_tensor(txt_mask) else None
        txt_mask_min = float(txt_mask.float().min().item()) if torch.is_tensor(txt_mask) else None
        txt_mask_max = float(txt_mask.float().max().item()) if torch.is_tensor(txt_mask) else None
        context_shape = _shape_of(context)
        clip_fea_shape = _shape_of(clip_fea)
        txt_byt5_shape = _shape_of(txt_byt5)
        context_len = int(context.shape[1]) if torch.is_tensor(context) and context.ndim >= 2 else None
        clip_fea_len = int(clip_fea.shape[1]) if torch.is_tensor(clip_fea) and clip_fea.ndim >= 2 else 0
        txt_byt5_len = int(txt_byt5.shape[1]) if torch.is_tensor(txt_byt5) and txt_byt5.ndim >= 2 else 0
        expected_pre_clip_txt_in_len = context_len
        expected_post_concat_txt_len = None if context_len is None else context_len + clip_fea_len + txt_byt5_len
        txt_mask_len = int(txt_mask.shape[-1]) if torch.is_tensor(txt_mask) and txt_mask.ndim >= 2 else None
        c_concat_channels = int(getattr(self, "_hy_last_apply_model_c_concat_channels", 0))
        noise_part = img
        c_concat_part = None
        if torch.is_tensor(img) and img.ndim >= 2 and c_concat_channels > 0 and img.shape[1] >= c_concat_channels:
            noise_part = img[:, :-c_concat_channels]
            c_concat_part = img[:, -c_concat_channels:]
        appears_preexpanded_for_clip = (
            txt_mask_len is not None
            and expected_pre_clip_txt_in_len is not None
            and expected_post_concat_txt_len is not None
            and txt_mask_len == expected_post_concat_txt_len
            and clip_fea_len > 0
        )
        if (
            _debug_enabled()
            and torch.is_tensor(img)
            and not getattr(self, "_hy_omniweaving_forward_orig_img_in_preview_logged", False)
        ):
            img_in_preview = self.img_in(img)
            self._hy_omniweaving_forward_orig_img_in_preview_logged = True
            _debug_log(
                "forward_orig img_in preview img_shape=%s img_norm=%s noise_shape=%s noise_norm=%s c_concat_shape=%s c_concat_norm=%s post_img_in_shape=%s post_img_in_norm=%s",
                _shape_of(img),
                _norm_of(img),
                _shape_of(noise_part),
                _norm_of(noise_part),
                _shape_of(c_concat_part),
                _norm_of(c_concat_part),
                _shape_of(img_in_preview),
                _norm_of(img_in_preview),
            )
        _debug_log(
            "forward_orig txt_mask shape=%s dtype=%s is_floating=%s will_apply_non_floating_conversion=%s nonzero=%s min=%s max=%s "
            "context_shape=%s txt_byt5_shape=%s clip_fea_shape=%s expected_txt_in_len=%s expected_post_concat_txt_len=%s "
            "txt_mask_len=%s appears_preexpanded_for_clip=%s",
            txt_mask_shape,
            txt_mask_dtype,
            txt_mask_is_floating,
            bool(torch.is_tensor(txt_mask) and not torch.is_floating_point(txt_mask)),
            txt_mask_nonzero,
            txt_mask_min,
            txt_mask_max,
            context_shape,
            txt_byt5_shape,
            clip_fea_shape,
            expected_pre_clip_txt_in_len,
            expected_post_concat_txt_len,
            txt_mask_len,
            appears_preexpanded_for_clip,
        )
        return forward_orig(*args, **kwargs)

    diffusion_model.forward_orig = types.MethodType(patched_forward_orig, diffusion_model)
    diffusion_model._hy_omniweaving_forward_orig_txt_mask_debug_patched = True
    logging.info("HY-OmniWeaving attached instance-local forward_orig txt_mask debug support.")
    return True


def ensure_hy_omniweaving_deepstack_support(model_patcher, sd: dict | None = None, mm_in_sd: dict | None = None):
    if mm_in_sd is None:
        mm_in_sd = extract_hy_omniweaving_mm_in_state_dict(sd or {})
    model = getattr(model_patcher, "model", None)
    if model is None:
        return False

    _ensure_hy_omniweaving_extra_conds_support(model)
    _ensure_hy_omniweaving_diffusion_wrapper(model_patcher)

    if len(mm_in_sd) == 0:
        return False

    diffusion_model = getattr(model, "diffusion_model", None)
    if diffusion_model is None:
        return False
    _ensure_hy_omniweaving_txt_mask_alignment_support(diffusion_model)
    _ensure_hy_omniweaving_forward_orig_txt_mask_debug_support(diffusion_model)
    if getattr(diffusion_model, "mm_in", None) is not None:
        return True

    linear_1_weight = mm_in_sd.get("linear_1.weight")
    linear_2_weight = mm_in_sd.get("linear_2.weight")
    if linear_1_weight is None or linear_2_weight is None:
        raise ValueError("HY-OmniWeaving mm_in weights are incomplete.")

    linear_cls = nn.Linear
    target_dtype = linear_1_weight.dtype
    target_device = linear_1_weight.device
    time_in = getattr(diffusion_model, "time_in", None)
    if time_in is not None:
        in_layer = getattr(time_in, "in_layer", None)
        if in_layer is not None:
            linear_cls = type(in_layer)
            weight = getattr(in_layer, "weight", None)
            if torch.is_tensor(weight):
                target_dtype = weight.dtype
                target_device = weight.device

    module = _TextProjection(
        in_channels=linear_1_weight.shape[1],
        hidden_size=linear_1_weight.shape[0],
        linear_cls=linear_cls,
        dtype=target_dtype,
        device=target_device,
    )
    missing, unexpected = module.load_state_dict(mm_in_sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        raise ValueError(
            f"Failed to attach HY-OmniWeaving mm_in cleanly. missing={missing} unexpected={unexpected}"
        )

    freeze_main = getattr(diffusion_model, "freeze_main", True)
    diffusion_model.mm_in = module
    diffusion_model.freeze_main = freeze_main
    diffusion_model._hy_omniweaving_mm_in_inactive = _is_effectively_zero(module.linear_2.weight) and _is_effectively_zero(module.linear_2.bias)
    if diffusion_model._hy_omniweaving_mm_in_inactive:
        logging.warning("HY-OmniWeaving attached mm_in, but linear_2 is all-zero so deepstack injection is numerically inactive.")
    source_linear_1_norm = _norm_of(linear_1_weight)
    source_linear_2_norm = _norm_of(linear_2_weight)
    attached_linear_1_norm = _norm_of(module.linear_1.weight)
    attached_linear_2_norm = _norm_of(module.linear_2.weight)
    _debug_log(
        "mm_in source_vs_attach source_linear1_norm=%.6f source_linear2_norm=%.6f attached_linear1_norm=%.6f attached_linear2_norm=%.6f",
        source_linear_1_norm if source_linear_1_norm is not None else -1.0,
        source_linear_2_norm if source_linear_2_norm is not None else -1.0,
        attached_linear_1_norm if attached_linear_1_norm is not None else -1.0,
        attached_linear_2_norm if attached_linear_2_norm is not None else -1.0,
    )
    return True


def _hy_omniweaving_diffusion_model_wrapper(executor, *args, **kwargs):
    transformer_options = args[-1] if len(args) > 0 and isinstance(args[-1], dict) else kwargs.get("transformer_options", {})
    if not isinstance(transformer_options, dict):
        transformer_options = {}

    all_stack_text_states = kwargs.pop("all_stack_text_states", None)
    diffusion_model = executor.class_obj
    if (
        all_stack_text_states is not None
        and getattr(diffusion_model, "mm_in", None) is not None
        and not getattr(diffusion_model, "_hy_omniweaving_mm_in_inactive", False)
    ):
        context = args[2] if len(args) > 2 else kwargs.get("context")
        projected = diffusion_model.mm_in(all_stack_text_states.to(dtype=context.dtype))
        patches_replace = dict(transformer_options.get("patches_replace", {}))
        dit_patches = dict(patches_replace.get("dit", {}))
        freeze_main = getattr(diffusion_model, "freeze_main", True)

        for index in range(min(len(projected), len(diffusion_model.double_blocks))):
            add = projected[index]
            previous = dit_patches.get(("double_block", index))

            def make_patch(add_tensor, previous_patch):
                def patch(args, extra):
                    out = previous_patch(args, extra) if previous_patch is not None else extra["original_block"](args)
                    txt = out["txt"]
                    n_slice = add_tensor.shape[-2]
                    if n_slice <= txt.shape[1]:
                        if freeze_main:
                            txt_front = txt[:, :-n_slice]
                            txt_back = txt[:, -n_slice:]
                            txt = torch.cat([txt_front, txt_back + add_tensor.detach()], dim=1)
                        else:
                            txt = txt.clone()
                            txt[:, -n_slice:] = txt[:, -n_slice:] + add_tensor
                        out["txt"] = txt
                    return out
                return patch

            dit_patches[("double_block", index)] = make_patch(add, previous)

        transformer_options = transformer_options.copy()
        patches_replace["dit"] = dit_patches
        transformer_options["patches_replace"] = patches_replace

        if not getattr(diffusion_model, "_hy_omniweaving_wrapper_logged", False):
            _debug_log(
                "diffusion wrapper fired all_stack_text_states_shape=%s all_stack_text_states_norm=%.6f projected_shape=%s projected_norm=%.6f patched_double_blocks=%s freeze_main=%s",
                _shape_of(all_stack_text_states),
                _norm_of(all_stack_text_states) if _norm_of(all_stack_text_states) is not None else -1.0,
                _shape_of(projected),
                _norm_of(projected) if _norm_of(projected) is not None else -1.0,
                min(len(projected), len(diffusion_model.double_blocks)),
                freeze_main,
            )
            diffusion_model._hy_omniweaving_wrapper_logged = True

        args = list(args)
        if len(args) > 0 and isinstance(args[-1], dict):
            args[-1] = transformer_options
        else:
            kwargs["transformer_options"] = transformer_options

    return executor(*args, **kwargs)


def _ensure_hy_omniweaving_diffusion_wrapper(model_patcher):
    import comfy.patcher_extension

    if not hasattr(model_patcher, "add_wrapper_with_key"):
        return False
    if getattr(model_patcher, "_hy_omniweaving_diffusion_wrapper_added", False):
        return False
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "hy_omniweaving_deepstack",
        _hy_omniweaving_diffusion_model_wrapper,
    )
    model_patcher._hy_omniweaving_diffusion_wrapper_added = True
    logging.info("HY-OmniWeaving attached instance-local diffusion wrapper for deepstack.")
    return True


def ensure_hy_omniweaving_text_encoder_support(clip):
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    if cond_stage_model is None:
        return False
    if hasattr(cond_stage_model, "_hy_omniweaving_text_encoder_patched"):
        return False

    def _find_template_end(tok_pairs, template_end=-1):
        count_im_start = 0
        if template_end != -1:
            return template_end
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if not torch.is_tensor(elem):
                if isinstance(elem, numbers.Integral):
                    if elem == 151644 and count_im_start < 2:
                        template_end = i
                        count_im_start += 1
        return template_end

    def _expanded_tok_pair_size(tok_pair):
        elem = tok_pair[0]
        if torch.is_tensor(elem):
            return int(elem.shape[0]) if elem.ndim > 0 else 1
        if isinstance(elem, dict):
            data = elem.get("data", None)
            if torch.is_tensor(data) and data.ndim > 0:
                return int(data.shape[0])
            return 1
        return 1

    def _resolve_crop_start(tok_pairs, explicit_crop_start, template_end, prepared_meta=None):
        if isinstance(prepared_meta, dict):
            prepared_crop_start = prepared_meta.get("crop_start", None)
            if prepared_crop_start is not None and int(prepared_crop_start) >= 0:
                return max(0, int(prepared_crop_start)), "prepared_meta"
        crop_start = explicit_crop_start
        source = "explicit"
        if crop_start is None or int(crop_start) < 0:
            crop_start = template_end
            source = "heuristic"
        if crop_start is None:
            crop_start = 0
        return max(0, int(crop_start)), source

    def _slice_seq_and_mask(seq, attention_mask, start_index):
        start_index = max(0, int(start_index))
        if torch.is_tensor(seq):
            seq = seq[..., start_index:, :] if seq.ndim == 4 else seq[:, start_index:]
        if torch.is_tensor(attention_mask):
            attention_mask = attention_mask[:, start_index:]
        return seq, attention_mask

    def _describe_attention_mask_state(value):
        if torch.is_tensor(value):
            return f"tensor{tuple(value.shape)}"
        if value is None:
            return "missing"
        return type(value).__name__

    def _collect_setclip_token_positions(tok_pairs, crop_start):
        expanded_index = 0
        token_positions = []
        image_metadata_positions = []
        for tok_pair in tok_pairs:
            elem = tok_pair[0]
            elem_size = _expanded_tok_pair_size(tok_pair)
            start = expanded_index
            end = start + elem_size
            if isinstance(elem, numbers.Integral) and int(elem) == 151653 and end > crop_start:
                token_positions.append(end)
            elif isinstance(elem, dict) and elem.get("original_type") == "image" and end > crop_start:
                image_metadata_positions.append(end)
            expanded_index = end
        return token_positions, image_metadata_positions

    def _find_setclip_start(tok_pairs, crop_start, prepared_meta=None):
        if isinstance(prepared_meta, dict) and bool(prepared_meta.get("used_fallback_text_only", False)):
            return 0, "prepared_text_only"
        token_positions, image_metadata_positions = _collect_setclip_token_positions(tok_pairs, crop_start)
        last_vision_end = token_positions[-1] if len(token_positions) > 0 else None
        last_image_end = image_metadata_positions[-1] if len(image_metadata_positions) > 0 else None
        prepared_roles = prepared_meta.get("ordered_roles", None) if isinstance(prepared_meta, dict) else None

        if last_vision_end is not None:
            source = "vision_end_token"
            if prepared_roles:
                source = f"prepared_roles+{source}"
            return max(0, last_vision_end - crop_start), source
        if last_image_end is not None:
            source = "image_metadata"
            if prepared_roles:
                source = f"prepared_roles+{source}"
            return max(0, last_image_end - crop_start), source
        return 0, "none"

    def _encode_deepstack(self, token_weight_pairs_qwen, crop_start, template_end):
        deepstack_layers = list(getattr(self, "deepstack_layers", []))
        if len(deepstack_layers) == 0:
            self._hy_last_qwen_attention_mask = None
            return None
        qwen_model = getattr(self, self.clip)
        self._hy_last_qwen_encode_source = _callable_debug_name(getattr(qwen_model, "encode_token_weights", None))
        qwen_model.reset_clip_options()
        qwen_model.set_clip_options({"layer": deepstack_layers})
        qwen_out, _, qwen_extra = qwen_model.encode_token_weights(token_weight_pairs_qwen)
        qwen_model.reset_clip_options()
        self._hy_last_qwen_extra_keys = sorted(qwen_extra.keys()) if isinstance(qwen_extra, dict) else []

        if qwen_out.ndim != 4:
            self._hy_last_qwen_attention_mask_state = "unexpected_qwen_out_rank"
            self._hy_last_qwen_attention_mask = None
            return None

        tok_pairs = token_weight_pairs_qwen[0]
        prepared_meta = getattr(self, "_hy_prepared_input_meta", None)
        effective_crop_start, crop_source = _resolve_crop_start(tok_pairs, crop_start, template_end, prepared_meta=prepared_meta)
        setclip_token_positions, image_metadata_positions = _collect_setclip_token_positions(tok_pairs, effective_crop_start)
        attention_mask = qwen_extra.get("attention_mask", None)
        qwen_out, attention_mask = _slice_seq_and_mask(qwen_out, attention_mask, effective_crop_start)
        self._hy_last_qwen_attention_mask_state = _describe_attention_mask_state(attention_mask)

        setclip_start = 0
        setclip_source = "disabled"
        if getattr(self, "setclip_output", False):
            setclip_start, setclip_source = _find_setclip_start(tok_pairs, effective_crop_start, prepared_meta=prepared_meta)
            if setclip_start > 0:
                qwen_out, attention_mask = _slice_seq_and_mask(qwen_out, attention_mask, setclip_start)
                self._hy_last_qwen_attention_mask_state = _describe_attention_mask_state(attention_mask)

        if torch.is_tensor(attention_mask):
            qwen_out = qwen_out * attention_mask.unsqueeze(1).unsqueeze(-1)
            self._hy_last_qwen_attention_mask = attention_mask.clone()
        else:
            self._hy_last_qwen_attention_mask = None
        _debug_log(
            "setclip token positions task=%s crop_start=%s ordered_roles=%s token_positions=%s image_metadata_positions=%s chosen=%s chosen_source=%s",
            getattr(self, "_hy_task_name", None),
            effective_crop_start,
            prepared_meta.get("ordered_roles", None) if isinstance(prepared_meta, dict) else None,
            setclip_token_positions,
            image_metadata_positions,
            setclip_start,
            setclip_source,
        )
        _debug_log(
            "deepstack encode task=%s qwen_class=%s qwen_encode=%s qwen_extra_keys=%s crop_start=%s crop_source=%s "
            "setclip=%s setclip_start=%s setclip_source=%s qwen_out_shape=%s attention_mask_shape=%s attention_mask_state=%s",
            getattr(self, "_hy_task_name", None),
            type(qwen_model).__name__,
            self._hy_last_qwen_encode_source,
            self._hy_last_qwen_extra_keys,
            effective_crop_start,
            crop_source,
            getattr(self, "setclip_output", False),
            setclip_start,
            setclip_source,
            _shape_of(qwen_out),
            _shape_of(attention_mask),
            self._hy_last_qwen_attention_mask_state,
        )
        return qwen_out.permute(1, 0, 2, 3).contiguous()

    orig_encode = cond_stage_model.encode_token_weights
    orig_set = cond_stage_model.set_clip_options
    orig_reset = cond_stage_model.reset_clip_options

    def patched_encode(self, token_weight_pairs):
        tok_pairs = token_weight_pairs["qwen25_7b"][0]
        template_end = -1
        if tok_pairs[0][0] == 27:
            if len(tok_pairs) > 36:
                template_end = 36

        cond, p, extra = orig_encode(token_weight_pairs)
        orig_extra_keys = sorted(extra.keys()) if isinstance(extra, dict) else []
        template_end = _find_template_end(tok_pairs, template_end)
        prepared_meta = getattr(self, "_hy_prepared_input_meta", None)
        effective_crop_start, crop_source = _resolve_crop_start(tok_pairs, getattr(self, "crop_start_output", None), template_end, prepared_meta=prepared_meta)
        setclip_token_positions, image_metadata_positions = _collect_setclip_token_positions(tok_pairs, effective_crop_start)
        attention_mask = extra.get("attention_mask", None)
        original_attention_mask = attention_mask
        original_mask_summary = _mask_summary(original_attention_mask)
        attention_mask_reason = "orig_encode_missing"
        if torch.is_tensor(attention_mask):
            attention_mask_reason = "orig_encode_tensor"
        elif attention_mask is not None:
            attention_mask_reason = f"orig_encode_non_tensor:{type(attention_mask).__name__}"

        setclip_start = 0
        setclip_source = "disabled"
        if self.setclip_output:
            setclip_start, setclip_source = _find_setclip_start(tok_pairs, effective_crop_start, prepared_meta=prepared_meta)
            if setclip_start > 0:
                cond, attention_mask = _slice_seq_and_mask(cond, attention_mask, setclip_start)
                if torch.is_tensor(attention_mask):
                    extra["attention_mask"] = attention_mask
                    attention_mask_reason = "setclip_tensor_retained"
                else:
                    extra.pop("attention_mask", None)
                    if original_attention_mask is None:
                        attention_mask_reason = "setclip_removed_missing_orig_encode_mask"
                    elif attention_mask is None:
                        attention_mask_reason = "setclip_removed_non_tensor_mask"
                    else:
                        attention_mask_reason = f"setclip_removed_non_tensor_mask:{type(attention_mask).__name__}"
        elif torch.is_tensor(attention_mask):
            attention_mask_reason = "setclip_disabled_tensor_unmodified"

        deepstack_hidden_states = _encode_deepstack(self, token_weight_pairs["qwen25_7b"], getattr(self, "crop_start_output", None), template_end)
        qwen_attention_mask = getattr(self, "_hy_last_qwen_attention_mask", None)
        qwen_mask_summary = _mask_summary(qwen_attention_mask)
        if (
            getattr(self, "setclip_output", False)
            and
            not torch.is_tensor(attention_mask)
            and torch.is_tensor(qwen_attention_mask)
            and torch.is_tensor(cond)
            and cond.ndim >= 2
            and qwen_attention_mask.ndim == 2
            and qwen_attention_mask.shape[0] == cond.shape[0]
            and qwen_attention_mask.shape[1] == cond.shape[1]
        ):
            attention_mask = qwen_attention_mask.to(device=cond.device)
            extra["attention_mask"] = attention_mask
            attention_mask_reason = "reconstructed_from_qwen_branch"
            _debug_log(
                "attention_mask reconstructed task=%s cond_stage_model_class=%s source=qwen_branch shape=%s "
                "orig_encode=%s qwen_encode=%s",
                getattr(self, "_hy_task_name", None),
                type(self).__name__,
                _shape_of(attention_mask),
                _callable_debug_name(orig_encode),
                getattr(self, "_hy_last_qwen_encode_source", "unset"),
            )
            _debug_log(
                "attention_mask reconstructed summary task=%s orig_mask=%s qwen_mask=%s reconstructed_mask=%s",
                getattr(self, "_hy_task_name", None),
                original_mask_summary,
                qwen_mask_summary,
                _mask_summary(attention_mask),
            )
        if deepstack_hidden_states is not None:
            extra["all_stack_text_states"] = deepstack_hidden_states
            if torch.is_tensor(cond) and cond.ndim >= 2 and deepstack_hidden_states.ndim >= 3:
                cond_tokens = cond.shape[1]
                deepstack_tokens = deepstack_hidden_states.shape[2]
                mask_tokens = attention_mask.shape[1] if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2 else None
                _debug_log(
                    "token alignment task=%s cond_tokens=%s deepstack_tokens=%s mask_tokens=%s crop_start=%s setclip_start=%s ordered_roles=%s",
                    getattr(self, "_hy_task_name", None),
                    cond_tokens,
                    deepstack_tokens,
                    mask_tokens,
                    effective_crop_start,
                    setclip_start,
                    prepared_meta.get("ordered_roles", None) if isinstance(prepared_meta, dict) else None,
                )
                if cond_tokens != deepstack_tokens:
                    logging.warning(
                        "HY-OmniWeaving token-length mismatch after crop/setclip: task=%s cond_tokens=%s deepstack_tokens=%s crop_start=%s setclip_start=%s",
                        getattr(self, "_hy_task_name", None),
                        cond_tokens,
                        deepstack_tokens,
                        effective_crop_start,
                        setclip_start,
                    )
        qwen_attention_mask_state = getattr(self, "_hy_last_qwen_attention_mask_state", "unset")
        if attention_mask_reason.startswith("setclip_removed") and qwen_attention_mask_state.startswith("tensor"):
            _debug_log(
                "attention_mask drop analysis task=%s cond_stage_model_class=%s orig_encode=%s orig_extra_keys=%s "
                "qwen_encode=%s qwen_extra_keys=%s qwen_attention_mask_state=%s "
                "inference=orig_encode returned no usable attention_mask before runtime-patch setclip handling",
                getattr(self, "_hy_task_name", None),
                type(self).__name__,
                _callable_debug_name(orig_encode),
                orig_extra_keys,
                getattr(self, "_hy_last_qwen_encode_source", "unset"),
                getattr(self, "_hy_last_qwen_extra_keys", []),
                qwen_attention_mask_state,
            )
        _debug_log(
            "attention_mask summary task=%s reason=%s orig_mask=%s qwen_mask=%s final_mask=%s",
            getattr(self, "_hy_task_name", None),
            attention_mask_reason,
            original_mask_summary,
            qwen_mask_summary,
            _mask_summary(extra.get("attention_mask")),
        )

        self.crop_start_source = crop_source
        self.setclip_start_source = setclip_source
        _debug_log(
            "patched_encode task=%s cond_stage_model_class=%s orig_encode=%s orig_extra_keys=%s crop_start=%s crop_source=%s "
            "visual_inputs=%s setclip=%s setclip_start=%s setclip_source=%s cond_shape=%s attention_mask_shape=%s attention_mask_reason=%s "
            "orig_attention_mask_state=%s final_attention_mask_state=%s extra_keys=%s deepstack_shape=%s",
            getattr(self, "_hy_task_name", None),
            type(self).__name__,
            _callable_debug_name(orig_encode),
            orig_extra_keys,
            effective_crop_start,
            crop_source,
            getattr(self, "_hy_visual_input_count", None),
            self.setclip_output,
            setclip_start,
            setclip_source,
            _shape_of(cond),
            _shape_of(extra.get("attention_mask")),
            attention_mask_reason,
            _describe_attention_mask_state(original_attention_mask),
            _describe_attention_mask_state(extra.get("attention_mask")),
            sorted(extra.keys()),
            _shape_of(extra.get("all_stack_text_states")),
        )
        _debug_log(
            "patched_encode setclip detail task=%s ordered_roles=%s token_positions=%s image_metadata_positions=%s chosen=%s chosen_source=%s",
            getattr(self, "_hy_task_name", None),
            prepared_meta.get("ordered_roles", None) if isinstance(prepared_meta, dict) else None,
            setclip_token_positions,
            image_metadata_positions,
            setclip_start,
            setclip_source,
        )
        return cond, p, extra

    def patched_set(self, options):
        orig_set(options)
        deepstack = options.get("deepstack", getattr(self, "deepstack_layers", []))
        if deepstack is None:
            deepstack = []
        self.deepstack_layers = list(deepstack)
        self.setclip_output = options.get("setclip", getattr(self, "setclip_output", False))
        crop_start = options.get("crop_start", getattr(self, "crop_start_output", None))
        self.crop_start_output = None if crop_start is None else int(crop_start)
        self.crop_start_source = "explicit" if crop_start is not None else "unset"
        self._hy_task_name = options.get("task_name", getattr(self, "_hy_task_name", None))
        self._hy_visual_input_count = options.get("visual_input_count", getattr(self, "_hy_visual_input_count", None))
        prepared_meta = options.get("prepared_meta", getattr(self, "_hy_prepared_input_meta", None))
        self._hy_prepared_input_meta = dict(prepared_meta) if isinstance(prepared_meta, dict) else None
        self.setclip_start_source = "pending" if self.setclip_output else "disabled"
        _debug_log(
            "patched_set task=%s crop_start=%s visual_inputs=%s setclip=%s prepared_meta=%s",
            self._hy_task_name,
            self.crop_start_output,
            self._hy_visual_input_count,
            self.setclip_output,
            self._hy_prepared_input_meta,
        )

    def patched_reset(self):
        orig_reset()
        self.deepstack_layers = []
        self.setclip_output = False
        self.crop_start_output = None
        self.crop_start_source = "unset"
        self._hy_task_name = None
        self._hy_visual_input_count = None
        self._hy_prepared_input_meta = None
        self.setclip_start_source = "unset"
        self._hy_last_qwen_attention_mask_state = "unset"
        self._hy_last_qwen_extra_keys = []
        self._hy_last_qwen_encode_source = "unset"
        self._hy_last_qwen_attention_mask = None

    cond_stage_model.deepstack_layers = []
    cond_stage_model.setclip_output = False
    cond_stage_model.crop_start_output = None
    cond_stage_model.crop_start_source = "unset"
    cond_stage_model._hy_task_name = None
    cond_stage_model._hy_visual_input_count = None
    cond_stage_model._hy_prepared_input_meta = None
    cond_stage_model.setclip_start_source = "unset"
    cond_stage_model._hy_last_qwen_attention_mask_state = "unset"
    cond_stage_model._hy_last_qwen_extra_keys = []
    cond_stage_model._hy_last_qwen_encode_source = "unset"
    cond_stage_model._hy_last_qwen_attention_mask = None
    cond_stage_model.encode_token_weights = types.MethodType(patched_encode, cond_stage_model)
    cond_stage_model.set_clip_options = types.MethodType(patched_set, cond_stage_model)
    cond_stage_model.reset_clip_options = types.MethodType(patched_reset, cond_stage_model)
    cond_stage_model._encode_deepstack = types.MethodType(_encode_deepstack, cond_stage_model)
    cond_stage_model._hy_omniweaving_text_encoder_patched = True
    logging.info(
        "HY-OmniWeaving attached instance-local text-encoder support for deepstack/setclip. "
        "cond_stage_model_class=%s orig_encode=%s qwen_class=%s qwen_encode=%s",
        type(cond_stage_model).__name__,
        _callable_debug_name(orig_encode),
        type(getattr(cond_stage_model, getattr(cond_stage_model, "clip", "qwen25_7b"), None)).__name__,
        _callable_debug_name(getattr(getattr(cond_stage_model, getattr(cond_stage_model, "clip", "qwen25_7b"), None), "encode_token_weights", None)),
    )
    return True


def _patch_qwen25_think_generation():
    import comfy
    import comfy.text_encoders.llama as llama

    if getattr(llama.Qwen25_7BVLI_Config, "stop_tokens", None) is None:
        llama.Qwen25_7BVLI_Config.stop_tokens = [151643, 151645]

    qwen25_cls = getattr(llama, "Qwen25_7BVLI", None)
    if qwen25_cls is not None and not getattr(qwen25_cls.__init__, "_hy_omniweaving_patched", False):
        original_qwen25_init = qwen25_cls.__init__

        def patched_qwen25_init(self, config_dict, dtype, device, operations):
            original_qwen25_init(self, config_dict, dtype, device, operations)
            model = getattr(self, "model", None)
            config = getattr(model, "config", None)
            if model is None or config is None or hasattr(model, "lm_head"):
                return

            hidden_size = getattr(config, "hidden_size", None)
            vocab_size = getattr(config, "vocab_size", None)
            if hidden_size is None or vocab_size is None:
                return

            linear_cls = getattr(operations, "Linear", nn.Linear)
            model.lm_head = linear_cls(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
            if not hasattr(model.lm_head, "comfy_cast_weights"):
                model.lm_head.comfy_cast_weights = False
            config.lm_head = True

        patched_qwen25_init._hy_omniweaving_patched = True
        llama.Qwen25_7BVLI.__init__ = patched_qwen25_init

    if not getattr(getattr(llama.BaseGenerate, "logits", None), "_hy_omniweaving_patched", False):
        def patched_logits(self, x):
            input = x[:, -1:]
            if hasattr(self.model, "lm_head"):
                module = self.model.lm_head
            else:
                module = self.model.embed_tokens

            offload_stream = None
            cast_bias_weight = getattr(comfy.ops, "cast_bias_weight", None)
            if getattr(module, "comfy_cast_weights", False) and callable(cast_bias_weight):
                weight, _, offload_stream = cast_bias_weight(module, input, offloadable=True)
            else:
                weight = module.weight.to(x)

            x = torch.nn.functional.linear(input, weight, None)

            uncast_bias_weight = getattr(comfy.ops, "uncast_bias_weight", None)
            if callable(uncast_bias_weight):
                uncast_bias_weight(module, weight, None, offload_stream)
            return x

        patched_logits._hy_omniweaving_patched = True
        llama.BaseGenerate.logits = patched_logits

    original_generate = llama.BaseGenerate.generate
    original_sample_token = getattr(llama.BaseGenerate, "sample_token", None)

    if original_sample_token is not None and not getattr(llama.BaseGenerate.sample_token, "_hy_omniweaving_patched", False):
        def patched_sample_token(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history, generator, do_sample=True, presence_penalty=0.0):
            suppressed_token_ids = getattr(self, "_hy_suppressed_token_ids", None)
            if suppressed_token_ids:
                logits = logits.clone()
                valid_ids = [int(token_id) for token_id in suppressed_token_ids if 0 <= int(token_id) < logits.shape[-1]]
                if len(valid_ids) > 0:
                    logits[:, valid_ids] = torch.finfo(logits.dtype).min
                    _debug_log(
                        "rewrite token suppression active count=%s first_ids=%s",
                        len(valid_ids),
                        valid_ids[:8],
                    )
            return original_sample_token(
                self,
                logits,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                token_history,
                generator,
                do_sample=do_sample,
                presence_penalty=presence_penalty,
            )

        patched_sample_token._hy_omniweaving_patched = True
        llama.BaseGenerate.sample_token = patched_sample_token

    if not getattr(llama.BaseGenerate.generate, "_hy_omniweaving_patched", False):
        def patched_generate(self, embeds=None, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0, seed=42, stop_tokens=None, initial_tokens=[], execution_dtype=None, min_tokens=0, presence_penalty=0.0):
            if stop_tokens is None:
                config = getattr(self, "model", None)
                config = getattr(config, "config", None)
                stop_tokens = getattr(config, "stop_tokens", None)
                if stop_tokens is None:
                    stop_tokens = [151643, 151645]
            return original_generate(
                self,
                embeds=embeds,
                do_sample=do_sample,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_tokens=stop_tokens,
                initial_tokens=initial_tokens,
                execution_dtype=execution_dtype,
                min_tokens=min_tokens,
                presence_penalty=presence_penalty,
            )

        patched_generate._hy_omniweaving_patched = True
        llama.BaseGenerate.generate = patched_generate
def _patch_autoencoder_legacy():
    import math
    import comfy
    import comfy.ldm.models.autoencoder as autoencoder

    if getattr(autoencoder.AutoencodingEngineLegacy.__init__, "_hy_omniweaving_patched", False):
        return

    original_init = autoencoder.AutoencodingEngineLegacy.__init__
    if "decoder_ddconfig" in original_init.__code__.co_varnames:
        return

    def patched_init(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        decoder_ddconfig = kwargs.pop("decoder_ddconfig", ddconfig)
        autoencoder.AutoencodingEngine.__init__(
            self,
            encoder_config={
                "target": "comfy.ldm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "comfy.ldm.modules.diffusionmodules.model.Decoder",
                "params": decoder_ddconfig,
            },
            **kwargs,
        )

        if ddconfig.get("conv3d", False):
            conv_op = comfy.ops.disable_weight_init.Conv3d
        else:
            conv_op = comfy.ops.disable_weight_init.Conv2d

        self.quant_conv = conv_op(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        self.post_quant_conv = conv_op(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if ddconfig.get("batch_norm_latent", False):
            self.bn_eps = 1e-4
            self.bn_momentum = 0.1
            self.ps = [2, 2]
            self.bn = torch.nn.BatchNorm2d(
                math.prod(self.ps) * ddconfig["z_channels"],
                eps=self.bn_eps,
                momentum=self.bn_momentum,
                affine=False,
                track_running_stats=True,
            )
            self.bn.eval()
        else:
            self.bn = None

    patched_init._hy_omniweaving_patched = True
    autoencoder.AutoencodingEngineLegacy.__init__ = patched_init

def apply_runtime_patches():
    return ensure_runtime_patches()


def ensure_runtime_patches():
    global _HY_OMNIWEAVING_RUNTIME_PATCHES_READY
    if _HY_OMNIWEAVING_RUNTIME_PATCHES_READY:
        return False
    _patch_qwen25_think_generation()
    _patch_autoencoder_legacy()
    _HY_OMNIWEAVING_RUNTIME_PATCHES_READY = True
    return True
