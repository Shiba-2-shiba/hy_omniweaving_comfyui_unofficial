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


def _norm_of(value):
    if torch.is_tensor(value):
        return float(value.float().norm().item())
    return None


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

    def patched_extra_conds(self, **kwargs):
        out = original_extra_conds(**kwargs)
        all_stack_text_states = kwargs.get("all_stack_text_states", None)
        if all_stack_text_states is not None:
            out["all_stack_text_states"] = _CONDDeepstackTextStates(all_stack_text_states)
        return out

    model.extra_conds = types.MethodType(patched_extra_conds, model)
    model._hy_omniweaving_extra_conds_patched = True
    logging.info("HY-OmniWeaving attached instance-local extra_conds support.")
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
    if getattr(diffusion_model, "mm_in", None) is not None:
        return True

    linear_1_weight = mm_in_sd.get("linear_1.weight")
    linear_2_weight = mm_in_sd.get("linear_2.weight")
    if linear_1_weight is None or linear_2_weight is None:
        raise ValueError("HY-OmniWeaving mm_in weights are incomplete.")

    linear_cls = nn.Linear
    time_in = getattr(diffusion_model, "time_in", None)
    if time_in is not None:
        in_layer = getattr(time_in, "in_layer", None)
        if in_layer is not None:
            linear_cls = type(in_layer)

    module = _TextProjection(
        in_channels=linear_1_weight.shape[1],
        hidden_size=linear_1_weight.shape[0],
        linear_cls=linear_cls,
        dtype=linear_1_weight.dtype,
        device=linear_1_weight.device,
    )
    missing, unexpected = module.load_state_dict(mm_in_sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        raise ValueError(
            f"Failed to attach HY-OmniWeaving mm_in cleanly. missing={missing} unexpected={unexpected}"
        )

    freeze_main = getattr(diffusion_model, "freeze_main", True)
    diffusion_model.mm_in = module
    diffusion_model.freeze_main = freeze_main
    _debug_log(
        "mm_in source_vs_attach source_linear1_norm=%.6f source_linear2_norm=%.6f attached_linear1_norm=%.6f attached_linear2_norm=%.6f",
        _norm_of(linear_1_weight) or -1.0,
        _norm_of(linear_2_weight) or -1.0,
        _norm_of(module.linear_1.weight) or -1.0,
        _norm_of(module.linear_2.weight) or -1.0,
    )
    return True


def _hy_omniweaving_diffusion_model_wrapper(executor, *args, **kwargs):
    transformer_options = args[-1] if len(args) > 0 and isinstance(args[-1], dict) else kwargs.get("transformer_options", {})
    if not isinstance(transformer_options, dict):
        transformer_options = {}

    all_stack_text_states = kwargs.pop("all_stack_text_states", None)
    diffusion_model = executor.class_obj
    if all_stack_text_states is not None and getattr(diffusion_model, "mm_in", None) is not None:
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
                _norm_of(all_stack_text_states) or -1.0,
                _shape_of(projected),
                _norm_of(projected) or -1.0,
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

    def _find_setclip_start(tok_pairs, template_end):
        extra_sizes = 0
        last_image_end = None
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if torch.is_tensor(elem) or isinstance(elem, numbers.Integral):
                continue
            if elem.get("original_type") == "image":
                elem_size = elem.get("data").shape[0]
                start = i + extra_sizes
                end = start + elem_size
                last_image_end = end
                extra_sizes += elem_size - 1
        if last_image_end is None:
            return 0
        return max(0, last_image_end - template_end)

    def _encode_deepstack(self, token_weight_pairs_qwen, template_end):
        deepstack_layers = list(getattr(self, "deepstack_layers", []))
        if len(deepstack_layers) == 0:
            return None
        qwen_model = getattr(self, self.clip)
        qwen_model.reset_clip_options()
        qwen_model.set_clip_options({"layer": deepstack_layers})
        qwen_out, _, qwen_extra = qwen_model.encode_token_weights(token_weight_pairs_qwen)
        qwen_model.reset_clip_options()

        if qwen_out.ndim != 4:
            return None

        qwen_out = qwen_out[:, :, template_end:]
        attention_mask = qwen_extra.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask[:, template_end:]

        if getattr(self, "setclip_output", False):
            setclip_start = _find_setclip_start(token_weight_pairs_qwen[0], template_end)
            if setclip_start > 0:
                qwen_out = qwen_out[:, :, setclip_start:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, setclip_start:]

        if attention_mask is not None:
            qwen_out = qwen_out * attention_mask.unsqueeze(1).unsqueeze(-1)
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
        template_end = _find_template_end(tok_pairs, template_end)

        if self.setclip_output:
            setclip_start = _find_setclip_start(tok_pairs, template_end)
            if setclip_start > 0:
                cond = cond[:, setclip_start:]
                attention_mask = extra.get("attention_mask", None)
                if attention_mask is not None:
                    extra["attention_mask"] = attention_mask[:, setclip_start:]

        deepstack_hidden_states = _encode_deepstack(self, token_weight_pairs["qwen25_7b"], template_end)
        if deepstack_hidden_states is not None:
            extra["all_stack_text_states"] = deepstack_hidden_states
        return cond, p, extra

    def patched_set(self, options):
        orig_set(options)
        deepstack = options.get("deepstack", getattr(self, "deepstack_layers", []))
        if deepstack is None:
            deepstack = []
        self.deepstack_layers = list(deepstack)
        self.setclip_output = options.get("setclip", getattr(self, "setclip_output", False))

    def patched_reset(self):
        orig_reset()
        self.deepstack_layers = []
        self.setclip_output = False

    cond_stage_model.deepstack_layers = []
    cond_stage_model.setclip_output = False
    cond_stage_model.encode_token_weights = types.MethodType(patched_encode, cond_stage_model)
    cond_stage_model.set_clip_options = types.MethodType(patched_set, cond_stage_model)
    cond_stage_model.reset_clip_options = types.MethodType(patched_reset, cond_stage_model)
    cond_stage_model._encode_deepstack = types.MethodType(_encode_deepstack, cond_stage_model)
    cond_stage_model._hy_omniweaving_text_encoder_patched = True
    logging.info("HY-OmniWeaving attached instance-local text-encoder support for deepstack/setclip.")
    return True


def _patch_qwen25_think_generation():
    import comfy.text_encoders.llama as llama

    if getattr(llama.Qwen25_7BVLI_Config, "stop_tokens", None) is None:
        llama.Qwen25_7BVLI_Config.stop_tokens = [151643, 151645]

    if getattr(llama.BaseGenerate.generate, "_hy_omniweaving_patched", False):
        return

    original_generate = llama.BaseGenerate.generate

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
    _patch_qwen25_think_generation()
    _patch_autoencoder_legacy()
