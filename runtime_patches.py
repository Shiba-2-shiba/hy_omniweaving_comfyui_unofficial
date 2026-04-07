"""
Runtime patch entrypoint for HY-OmniWeaving support.

Current state:
- Some parity support still exists in the ComfyUI core checkout.
- This module progressively absorbs the easier pieces behind custom-node-owned
  monkey patches so that future refactors have a single home.
- The hardest remaining dependency is the HunyuanVideo transformer deepstack
  integration inside `comfy/ldm/hunyuan_video/model.py`.
"""

from __future__ import annotations
import numbers

import torch
from torch import nn


def _patch_hunyuan_image_te():
    import comfy.text_encoders.hunyuan_image as hunyuan_image

    if hasattr(hunyuan_image.HunyuanImageTEModel, "_encode_deepstack"):
        return

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

    orig_init = hunyuan_image.HunyuanImageTEModel.__init__
    orig_encode = hunyuan_image.HunyuanImageTEModel.encode_token_weights
    orig_set = hunyuan_image.HunyuanImageTEModel.set_clip_options
    orig_reset = hunyuan_image.HunyuanImageTEModel.reset_clip_options

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.deepstack_layers = []
        self.setclip_output = False

    def patched_encode(self, token_weight_pairs):
        tok_pairs = token_weight_pairs["qwen25_7b"][0]
        template_end = -1
        if tok_pairs[0][0] == 27:
            if len(tok_pairs) > 36:
                template_end = 36

        cond, p, extra = orig_encode(self, token_weight_pairs)
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
        orig_set(self, options)
        deepstack = options.get("deepstack", getattr(self, "deepstack_layers", []))
        if deepstack is None:
            deepstack = []
        self.deepstack_layers = list(deepstack)
        self.setclip_output = options.get("setclip", getattr(self, "setclip_output", False))

    def patched_reset(self):
        orig_reset(self)
        self.deepstack_layers = []
        self.setclip_output = False

    hunyuan_image.HunyuanImageTEModel.__init__ = patched_init
    hunyuan_image.HunyuanImageTEModel.encode_token_weights = patched_encode
    hunyuan_image.HunyuanImageTEModel.set_clip_options = patched_set
    hunyuan_image.HunyuanImageTEModel.reset_clip_options = patched_reset
    hunyuan_image.HunyuanImageTEModel._encode_deepstack = _encode_deepstack


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


def _patch_model_detection():
    import comfy.model_detection as model_detection

    if getattr(model_detection.detect_unet_config, "_hy_omniweaving_patched", False):
        return

    original = model_detection.detect_unet_config

    def patched_detect_unet_config(state_dict, key_prefix, metadata=None):
        dit_config = original(state_dict, key_prefix, metadata=metadata)
        if isinstance(dit_config, dict) and dit_config.get("image_model") == "hunyuan_video":
            dit_config["deepstack"] = f"{key_prefix}mm_in.linear_1.weight" in state_dict
        return dit_config

    patched_detect_unet_config._hy_omniweaving_patched = True
    model_detection.detect_unet_config = patched_detect_unet_config


def _patch_model_base():
    import comfy.model_base as model_base
    import comfy.conds

    if getattr(model_base.HunyuanVideo15.extra_conds, "_hy_omniweaving_patched", False):
        return

    original = model_base.HunyuanVideo15.extra_conds

    def patched_extra_conds(self, **kwargs):
        out = original(self, **kwargs)
        all_stack_text_states = kwargs.get("all_stack_text_states", None)
        if all_stack_text_states is not None:
            out["all_stack_text_states"] = comfy.conds.CONDRegular(all_stack_text_states)
        return out

    patched_extra_conds._hy_omniweaving_patched = True
    model_base.HunyuanVideo15.extra_conds = patched_extra_conds

    if hasattr(model_base, "HunyuanVideo15_SR_Distilled"):
        original_sr = model_base.HunyuanVideo15_SR_Distilled.extra_conds

        def patched_sr_extra_conds(self, **kwargs):
            out = original_sr(self, **kwargs)
            all_stack_text_states = kwargs.get("all_stack_text_states", None)
            if all_stack_text_states is not None:
                out["all_stack_text_states"] = comfy.conds.CONDRegular(all_stack_text_states)
            return out

        patched_sr_extra_conds._hy_omniweaving_patched = True
        model_base.HunyuanVideo15_SR_Distilled.extra_conds = patched_sr_extra_conds


def _patch_hunyuan_video_model():
    import comfy.ldm.hunyuan_video.model as hv_model

    if getattr(hv_model.HunyuanVideo, "_hy_omniweaving_runtime_patched", False):
        return

    class _TextProjection(nn.Module):
        def __init__(self, in_channels, hidden_size, dtype=None, device=None, operations=None):
            super().__init__()
            self.linear_1 = operations.Linear(in_channels, hidden_size, bias=True, dtype=dtype, device=device)
            self.act_1 = nn.SiLU()
            self.linear_2 = operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device)

        def forward(self, caption):
            hidden_states = self.linear_1(caption)
            hidden_states = self.act_1(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            return hidden_states

    original_init = hv_model.HunyuanVideo.__init__
    original_forward = hv_model.HunyuanVideo._forward

    def patched_init(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        deepstack = kwargs.pop("deepstack", False)
        original_init(self, image_model=image_model, final_layer=final_layer, dtype=dtype, device=device, operations=operations, **kwargs)
        if not hasattr(self, "freeze_main"):
            self.freeze_main = True
        if deepstack and getattr(self, "mm_in", None) is None and operations is not None:
            self.mm_in = _TextProjection(self.params.context_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        elif getattr(self, "mm_in", None) is None:
            self.mm_in = None

    def patched_forward(self, x, timestep, context, y=None, txt_byt5=None, clip_fea=None, guidance=None, attention_mask=None, guiding_frame_index=None, ref_latent=None, disable_time_r=False, control=None, transformer_options={}, **kwargs):
        all_stack_text_states = kwargs.pop("all_stack_text_states", None)
        if all_stack_text_states is not None and getattr(self, "mm_in", None) is not None:
            projected = self.mm_in(all_stack_text_states.to(dtype=context.dtype))
            patches_replace = dict(transformer_options.get("patches_replace", {}))
            dit_patches = dict(patches_replace.get("dit", {}))
            freeze_main = getattr(self, "freeze_main", True)

            for index in range(min(len(projected), len(self.double_blocks))):
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

        return original_forward(
            self,
            x,
            timestep,
            context,
            y=y,
            txt_byt5=txt_byt5,
            clip_fea=clip_fea,
            guidance=guidance,
            attention_mask=attention_mask,
            guiding_frame_index=guiding_frame_index,
            ref_latent=ref_latent,
            disable_time_r=disable_time_r,
            control=control,
            transformer_options=transformer_options,
            **kwargs,
        )

    hv_model.HunyuanVideo.__init__ = patched_init
    hv_model.HunyuanVideo._forward = patched_forward
    hv_model.HunyuanVideo._hy_omniweaving_runtime_patched = True


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

def _core_has_minimum_deepstack_support() -> bool:
    import comfy.ldm.hunyuan_video.model as hv_model
    import comfy.model_base as model_base
    import comfy.text_encoders.hunyuan_image as hunyuan_image

    model_ok = getattr(hv_model.HunyuanVideo, "_hy_omniweaving_runtime_patched", False) or (
        hasattr(hv_model.HunyuanVideoParams, "__dataclass_fields__") and "deepstack" in hv_model.HunyuanVideoParams.__dataclass_fields__
    )
    extra_conds_ok = "all_stack_text_states" in model_base.HunyuanVideo15.extra_conds.__code__.co_consts
    text_ok = hasattr(hunyuan_image.HunyuanImageTEModel, "_encode_deepstack")
    return model_ok and extra_conds_ok and text_ok


def apply_runtime_patches():
    _patch_hunyuan_image_te()
    _patch_qwen25_think_generation()
    _patch_model_detection()
    _patch_model_base()
    _patch_hunyuan_video_model()
    _patch_autoencoder_legacy()
    if _core_has_minimum_deepstack_support():
        return
    raise RuntimeError(
        "hy_omniweaving_comfyui failed to establish the minimum HY-OmniWeaving parity hooks through runtime patching."
    )
