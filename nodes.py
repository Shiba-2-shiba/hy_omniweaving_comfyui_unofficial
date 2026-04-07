import json
import logging
import math
import node_helpers
import os
import torch
import comfy.clip_vision
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


def _clip_has_byt5_branch(clip) -> bool:
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    return getattr(cond_stage_model, "byt5_small", None) is not None


def _text_encoder_options():
    return folder_paths.get_filename_list("text_encoders")


def _preferred_text_encoder_name(*preferred_names: str, contains: str | None = None) -> str | None:
    options = _text_encoder_options()
    if len(options) == 0:
        return None
    option_set = set(options)
    for name in preferred_names:
        if name in option_set:
            return name
    if contains is not None:
        lowered = contains.lower()
        for option in options:
            if lowered in option.lower():
                return option
    return options[0]


def _unwrap_text_encoder_state_dict(sd):
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    return sd


def _normalize_hy_omniweaving_text_encoder_state_dict(sd: dict) -> dict:
    sd = _unwrap_text_encoder_state_dict(sd)
    if not isinstance(sd, dict):
        raise TypeError("Expected a text encoder state_dict mapping.")

    normalized = {k: v for k, v in sd.items() if k != "__metadata__"}
    if "model.language_model.layers.0.self_attn.k_proj.weight" in normalized:
        normalized = comfy.utils.state_dict_prefix_replace(
            normalized,
            {
                "model.language_model.": "model.",
                "model.visual.": "visual.",
                "final_layer_norm.": "model.norm.",
            },
        )
    return normalized


def _load_hy_omniweaving_dual_text_encoder(qwen_text_encoder: str, byt5_text_encoder: str, device: str = "default"):
    qwen_path = folder_paths.get_full_path_or_raise("text_encoders", qwen_text_encoder)
    byt5_path = folder_paths.get_full_path_or_raise("text_encoders", byt5_text_encoder)

    qwen_sd = _normalize_hy_omniweaving_text_encoder_state_dict(
        comfy.utils.load_torch_file(qwen_path, safe_load=True)
    )
    byt5_sd = comfy.utils.load_torch_file(byt5_path, safe_load=True)

    model_options = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts=[qwen_sd, byt5_sd],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.HUNYUAN_VIDEO_15,
        model_options=model_options,
    )
    logging.info(
        "HYOmniWeavingTextEncoderLoader loaded dual text encoders: qwen=%s byt5=%s",
        qwen_text_encoder,
        byt5_text_encoder,
    )
    return clip


def _convert_split_hy_omniweaving_attention_qkv(sd: dict, strict_mode: bool = True):
    converted = 0
    seen_partial = []
    block_indices = set()

    for key in sd.keys():
        if not key.startswith("double_blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        if parts[2] in ("img_attn_q", "img_attn_k", "img_attn_v", "txt_attn_q", "txt_attn_k", "txt_attn_v"):
            try:
                block_indices.add(int(parts[1]))
            except ValueError:
                pass

    for idx in sorted(block_indices):
        for attn_prefix in ("img_attn", "txt_attn"):
            for end in ("weight", "bias"):
                qkv_key = f"double_blocks.{idx}.{attn_prefix}.qkv.{end}"
                if qkv_key in sd:
                    continue

                q_key = f"double_blocks.{idx}.{attn_prefix}_q.{end}"
                k_key = f"double_blocks.{idx}.{attn_prefix}_k.{end}"
                v_key = f"double_blocks.{idx}.{attn_prefix}_v.{end}"
                present = [k for k in (q_key, k_key, v_key) if k in sd]

                if len(present) == 0:
                    continue

                if len(present) != 3:
                    seen_partial.append((idx, attn_prefix, end, tuple(present)))
                    continue

                sd[qkv_key] = torch.cat((sd.pop(q_key), sd.pop(k_key), sd.pop(v_key)), dim=0)
                converted += 1

    if strict_mode and len(seen_partial) > 0:
        raise ValueError(f"Partial HY-OmniWeaving split attention tensors found: {seen_partial}")

    return sd, converted, seen_partial


def _build_decoder_ddconfig_if_needed(sd: dict, ddconfig: dict):
    decoder_ch = sd["decoder.conv_in.weight"].shape[0] // ddconfig["ch_mult"][-1]
    if decoder_ch != ddconfig["ch"]:
        decoder_ddconfig = ddconfig.copy()
        decoder_ddconfig["ch"] = decoder_ch
        return decoder_ddconfig
    return None


def _is_omniweaving_vae_state_dict(sd: dict) -> bool:
    return "decoder.conv_in.conv.weight" in sd and "encoder.conv_in.conv.weight" in sd


def _load_omniweaving_vae_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "omniweaving_vae_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _instantiate_omniweaving_vae_model(config: dict):
    try:
        from .omniweaving_vae import OmniWeavingAutoencoderKLConv3D
    except ImportError:
        from omniweaving_vae import OmniWeavingAutoencoderKLConv3D

    return OmniWeavingAutoencoderKLConv3D(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        latent_channels=config["latent_channels"],
        block_out_channels=config["block_out_channels"],
        layers_per_block=config["layers_per_block"],
        ffactor_spatial=config["ffactor_spatial"],
        ffactor_temporal=config["ffactor_temporal"],
        sample_size=config["sample_size"],
        sample_tsize=config["sample_tsize"],
        scaling_factor=config.get("scaling_factor"),
        shift_factor=config.get("shift_factor"),
        downsample_match_channel=config.get("downsample_match_channel", True),
        upsample_match_channel=config.get("upsample_match_channel", True),
    )


def _filter_known_optional_vae_missing_keys(missing_keys):
    optional_suffixes = {
        "encoder.mid.block_1.temb_proj.weight",
        "encoder.mid.block_1.temb_proj.bias",
        "encoder.mid.block_2.temb_proj.weight",
        "encoder.mid.block_2.temb_proj.bias",
        "decoder.mid.block_1.temb_proj.weight",
        "decoder.mid.block_1.temb_proj.bias",
        "decoder.mid.block_2.temb_proj.weight",
        "decoder.mid.block_2.temb_proj.bias",
    }
    filtered = [key for key in missing_keys if key not in optional_suffixes]
    ignored = [key for key in missing_keys if key in optional_suffixes]
    return filtered, ignored


class HYOmniWeavingVAE(comfy.sd.VAE):
    def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None):
        if config is not None or sd is None or "decoder.conv_in.weight" not in sd:
            if config is None and sd is not None and _is_omniweaving_vae_state_dict(sd):
                self._init_omniweaving_vae(sd=sd, device=device, dtype=dtype)
                return
            super().__init__(sd=sd, device=device, config=config, dtype=dtype, metadata=metadata)
            return

        if "decoder.up_blocks.0.resnets.0.norm1.weight" in sd.keys():
            sd = comfy.sd.diffusers_convert.convert_vae_state_dict(sd)

        if sd["decoder.conv_in.weight"].shape[1] == 64:
            super().__init__(sd=sd, device=device, config=config, dtype=dtype, metadata=metadata)
            return
        if sd["decoder.conv_in.weight"].shape[1] == 32 and sd["decoder.conv_in.weight"].ndim == 5:
            super().__init__(sd=sd, device=device, config=config, dtype=dtype, metadata=metadata)
            return

        if comfy.model_management.is_amd():
            vae_kl_mem_ratio = 2.73
        else:
            vae_kl_mem_ratio = 1.0

        self.memory_used_encode = lambda shape, dtype_: (1767 * shape[2] * shape[3]) * comfy.model_management.dtype_size(dtype_) * vae_kl_mem_ratio
        self.memory_used_decode = lambda shape, dtype_: (2178 * shape[2] * shape[3] * 64) * comfy.model_management.dtype_size(dtype_) * vae_kl_mem_ratio
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.latent_dim = 2
        self.output_channels = 3
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = False
        self.not_video = False
        self.size = None
        self.downscale_index_formula = None
        self.upscale_index_formula = None
        self.extra_1d_channel = None
        self.crop_input = True
        self.audio_sample_rate = 44100

        ddconfig = {"double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3, "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0}

        if "encoder.down.2.downsample.conv.weight" not in sd and "decoder.up.3.upsample.conv.weight" not in sd:
            ddconfig["ch_mult"] = [1, 2, 4]
            self.downscale_ratio = 4
            self.upscale_ratio = 4

        self.latent_channels = ddconfig["z_channels"] = sd["decoder.conv_in.weight"].shape[1]
        if "decoder.post_quant_conv.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"decoder.post_quant_conv.": "post_quant_conv.", "encoder.quant_conv.": "quant_conv."})

        if "bn.running_mean" in sd:
            ddconfig["batch_norm_latent"] = True
            self.downscale_ratio *= 2
            self.upscale_ratio *= 2
            self.latent_channels *= 4
            old_memory_used_decode = self.memory_used_decode
            self.memory_used_decode = lambda shape, dtype_: old_memory_used_decode(shape, dtype_) * 4.0

        decoder_ddconfig = _build_decoder_ddconfig_if_needed(sd, ddconfig)

        if "post_quant_conv.weight" in sd:
            self.first_stage_model = comfy.sd.AutoencoderKL(
                ddconfig=ddconfig,
                embed_dim=sd["post_quant_conv.weight"].shape[1],
                **({"decoder_ddconfig": decoder_ddconfig} if decoder_ddconfig is not None else {}),
            )
        else:
            self.first_stage_model = comfy.sd.AutoencodingEngine(
                regularizer_config={"target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                encoder_config={"target": "comfy.ldm.modules.diffusionmodules.model.Encoder", "params": ddconfig},
                decoder_config={"target": "comfy.ldm.modules.diffusionmodules.model.Decoder", "params": decoder_ddconfig if decoder_ddconfig is not None else ddconfig},
            )

        self.first_stage_model = self.first_stage_model.eval()
        self.device = device if device is not None else comfy.model_management.vae_device()
        offload_device = comfy.model_management.vae_offload_device()
        if dtype is None:
            dtype = comfy.model_management.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        comfy.model_management.archive_model_dtypes(self.first_stage_model)
        self.output_device = comfy.model_management.intermediate_device()
        mp = comfy.model_patcher.CoreModelPatcher
        if self.disable_offload:
            mp = comfy.model_patcher.ModelPatcher
        self.patcher = mp(self.first_stage_model, load_device=self.device, offload_device=offload_device)

        m, u = self.first_stage_model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())
        m, ignored_missing = _filter_known_optional_vae_missing_keys(m)
        if len(ignored_missing) > 0:
            logging.info("Ignoring known optional HY-OmniWeaving VAE keys %s", ignored_missing)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))
        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        logging.info("VAE load device: {}, offload device: {}, dtype: {}".format(self.device, offload_device, self.vae_dtype))
        self.model_size()

    def _init_omniweaving_vae(self, sd: dict, device=None, dtype=None):
        config = _load_omniweaving_vae_config()
        ffactor_spatial = config["ffactor_spatial"]
        ffactor_temporal = config["ffactor_temporal"]

        self.memory_used_encode = lambda shape, dtype_: (2800 * shape[-2] * shape[-1]) * comfy.model_management.dtype_size(dtype_)
        self.memory_used_decode = lambda shape, dtype_: (
            2800 * shape[-3] * shape[-2] * shape[-1] * ffactor_spatial * ffactor_spatial
        ) * comfy.model_management.dtype_size(dtype_)
        self.downscale_ratio = (lambda a: max(0, math.floor((a + ffactor_temporal - 1) / ffactor_temporal)), ffactor_spatial, ffactor_spatial)
        self.downscale_index_formula = (ffactor_temporal, ffactor_spatial, ffactor_spatial)
        self.upscale_ratio = (lambda a: max(0, a * ffactor_temporal - (ffactor_temporal - 1)), ffactor_spatial, ffactor_spatial)
        self.upscale_index_formula = (ffactor_temporal, ffactor_spatial, ffactor_spatial)
        self.latent_channels = config["latent_channels"]
        self.latent_dim = 3
        self.output_channels = config["out_channels"]
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
        self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        self.disable_offload = False
        self.not_video = False
        self.size = None
        self.extra_1d_channel = None
        self.crop_input = True
        self.audio_sample_rate = 44100

        self.first_stage_model = _instantiate_omniweaving_vae_model(config).eval()
        self.device = device if device is not None else comfy.model_management.vae_device()
        offload_device = comfy.model_management.vae_offload_device()
        if dtype is None:
            dtype = comfy.model_management.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        comfy.model_management.archive_model_dtypes(self.first_stage_model)
        self.output_device = comfy.model_management.intermediate_device()
        mp = comfy.model_patcher.CoreModelPatcher
        if self.disable_offload:
            mp = comfy.model_patcher.ModelPatcher
        self.patcher = mp(self.first_stage_model, load_device=self.device, offload_device=offload_device)

        m, u = self.first_stage_model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())
        m, ignored_missing = _filter_known_optional_vae_missing_keys(m)
        if len(ignored_missing) > 0:
            logging.info("Ignoring known optional HY-OmniWeaving VAE keys %s", ignored_missing)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))
        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        logging.info(
            "HY-OmniWeaving VAE load device: %s, offload device: %s, dtype: %s",
            self.device,
            offload_device,
            self.vae_dtype,
        )
        self.model_size()


class HYOmniWeavingTextEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        text_encoder_options = _text_encoder_options()
        return io.Schema(
            node_id="HYOmniWeavingTextEncoderLoader",
            display_name="HY OmniWeaving Text Encoder Loader",
            category="advanced/loaders",
            description="Loads the OmniWeaving fine-tuned Qwen2.5-VL checkpoint together with the ByT5 checkpoint as a HunyuanVideo 1.5 dual text encoder.",
            inputs=[
                io.Combo.Input(
                    "qwen_text_encoder",
                    options=text_encoder_options,
                    default=_preferred_text_encoder_name("qwen_2.5_vl_7b_finetuned_model.safetensors", "qwen_2.5_vl_7b.safetensors"),
                ),
                io.Combo.Input(
                    "byt5_text_encoder",
                    options=text_encoder_options,
                    default=_preferred_text_encoder_name(contains="byt5"),
                ),
                io.Combo.Input("device", options=["default", "cpu"], default="default", advanced=True),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, qwen_text_encoder, byt5_text_encoder, device="default") -> io.NodeOutput:
        clip = _load_hy_omniweaving_dual_text_encoder(qwen_text_encoder, byt5_text_encoder, device=device)
        return io.NodeOutput(clip)


class HYOmniWeavingUNetLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYOmniWeavingUNetLoader",
            display_name="HY OmniWeaving UNet Loader",
            category="advanced/loaders",
            description="Loads HY-OmniWeaving/HunyuanVideo 1.5 transformers and converts split q/k/v attention tensors into the qkv layout expected by stock ComfyUI.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="default"),
                io.Boolean.Input("strict_mode", default=True, advanced=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype, strict_mode=True) -> io.NodeOutput:
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        sd, converted, partial = _convert_split_hy_omniweaving_attention_qkv(sd, strict_mode=strict_mode)
        if converted > 0:
            logging.info(f"HYOmniWeavingUNetLoader converted {converted} split attention tensors to qkv format.")
        if len(partial) > 0 and not strict_mode:
            logging.warning(f"HYOmniWeavingUNetLoader encountered partial split attention tensors: {partial}")
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        if model is None:
            raise RuntimeError(f"Failed to load HY-OmniWeaving diffusion model: {unet_name}")
        return io.NodeOutput(model)


class HYOmniWeavingVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYOmniWeavingVAELoader",
            display_name="HY OmniWeaving VAE Loader",
            category="loaders",
            description="Loads HY-OmniWeaving/HunyuanVideo VAE files using a decoder-aware config path compatible with the fork behavior.",
            inputs=[
                io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae")),
            ],
            outputs=[
                io.Vae.Output(),
            ],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = HYOmniWeavingVAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return io.NodeOutput(vae)


class TextEncodeHunyuanVideo15Omni(io.ComfyNode):
    TASK_SYSTEM_PROMPTS = {
        "t2v": "You are a helpful assistant. Describe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video.",
        "i2v": "You are a helpful assistant. Given a text instruction and an input image, you need to explain how the user's text instruction should alter the image to introduce motion and evolution over time. Generate a video using this image as the first frame that meets the user's requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency.",
        "reference2v": "You are a helpful assistant. Given a text instruction and one or more input images, you need to explain how to extract and combine key information from the input images to construct a new image as the video's first frame, and then how the user's text instruction should alter the image to introduce motion and evolution over time. Generate a video that meets the user's requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency.",
        "interpolation": "You are a helpful assistant. Given a text instruction, an image as the first frame of the video, and another image as the last frame of the video, you need to analyze the visual trajectory required to transition from the start to the end. Determine how the elements in the first frame must evolve, move, or transform to align with the last frame based on the text instruction. Generate a video that seamlessly connects these two frames, ensuring the motion and evolution between them fulfill the text description while maintaining temporal consistency",
        "editing": "You are a helpful assistant. Given a text instruction and an input video, you need to analyze the visual content and temporal dynamics of the input video, and then explain how the user's text instruction should modify the video's visual style, objects, or scene composition. Generate an edited video that meets the user's requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence.",
        "tiv2v": "You are a helpful assistant. Given a text instruction, a reference image and an input video, you need to analyze the visual content and temporal dynamics of the input video, alongside the scene or subject characteristics of the reference image. Explain how the user's text instruction directs the application of the reference image's visual attributes onto the input video. Generate an edited video that meets the user's requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence.",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYOmniWeavingTextEncode",
            display_name="HY OmniWeaving Text Encode",
            category="conditioning/video_models",
            description="HY-OmniWeaving-oriented HunyuanVideo 1.5 text encoder path with parity guards for visual inputs, optional think-mode prompt expansion, and deepstack/setclip options.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("task", options=["t2v", "i2v", "interpolation", "reference2v", "editing", "tiv2v"], default="t2v"),
                io.Boolean.Input("use_visual_inputs", default=True, advanced=True),
                io.Int.Input("max_visual_inputs", default=8, min=1, max=64, advanced=True),
                io.Boolean.Input("think", default=False, advanced=True),
                io.Int.Input("think_max_new_tokens", default=1000, min=1, max=4096, advanced=True),
                io.String.Input("deepstack_layers", default="8,16,24", advanced=True),
                io.Boolean.Input("setclip", default=True, advanced=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @staticmethod
    def _task_system_prompt(task: str) -> str:
        return TextEncodeHunyuanVideo15Omni.TASK_SYSTEM_PROMPTS.get(task, TextEncodeHunyuanVideo15Omni.TASK_SYSTEM_PROMPTS["t2v"])

    @classmethod
    def _build_template(cls, task: str, image_count: int, add_generation_prompt: bool = False) -> str:
        system_prompt = cls._task_system_prompt(task)
        visual_tokens = "<|vision_start|><|image_pad|><|vision_end|>\n" * image_count
        template = (
            "<|im_start|>system\n"
            f"{system_prompt}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{visual_tokens}" + "{}<|im_end|>\n"
        )
        if add_generation_prompt:
            template += "<|im_start|>assistant\n"
        return template

    @classmethod
    def _build_think_template(cls, task: str, image_count: int) -> str:
        return cls._build_template(task, image_count, add_generation_prompt=True)

    @staticmethod
    def _tokenize_with_template(clip, text, template, image_embeds):
        try:
            return clip.tokenize(text, llama_template=template, images=image_embeds)
        except TypeError:
            embeds = None
            if len(image_embeds) > 0:
                embeds = torch.stack(image_embeds, dim=0)
            return clip.tokenize(text, llama_template=template, image_embeds=embeds, image_interleave=1)

    @staticmethod
    def _extract_image_embeds(clip_vision_output, max_visual_inputs: int):
        if clip_vision_output is None:
            return []
        mm_projected = getattr(clip_vision_output, "mm_projected", None)
        if mm_projected is None:
            return []
        if mm_projected.ndim == 2:
            return [mm_projected]
        count = min(mm_projected.shape[0], max_visual_inputs)
        return [mm_projected[i] for i in range(count)]

    @staticmethod
    def _require_full_text_path(clip):
        if _clip_has_byt5_branch(clip):
            return
        raise ValueError(
            "HY-OmniWeaving requires the ByT5-enabled text path for parity. "
            "Use a loader route that preserves both Qwen2.5-VL and ByT5/Glyph conditioning."
        )

    @staticmethod
    def _require_visual_inputs(task: str, use_visual_inputs: bool, clip_vision_output):
        if not use_visual_inputs:
            return
        if task not in ("i2v", "interpolation", "reference2v", "editing", "tiv2v"):
            return
        if clip_vision_output is not None:
            return
        raise ValueError(
            f"Task '{task}' with use_visual_inputs=True requires clip_vision_output for HY-OmniWeaving parity."
        )

    @classmethod
    def _rewrite_prompt_with_think(cls, clip, prompt, task, image_embeds, max_new_tokens: int) -> str:
        if task not in ("t2v", "i2v", "interpolation"):
            raise ValueError("Think mode is currently intended only for t2v, i2v, or interpolation tasks.")

        if task == "i2v":
            expand_prefix = "Here is a concise description of the target video starting with the given image: "
            expand_postfix = " Please generate a more detailed description based on the provided image and the short description."
        elif task == "interpolation":
            expand_prefix = "Here is a concise description of how the video transitions from the first image to the second image: "
            expand_postfix = " Please generate a more detailed description of the transition, based on the provided images and the short description."
        else:
            expand_prefix = "Here is a concise description of the target video: "
            expand_postfix = " Please generate a more detailed description based on the short description."

        think_prompt = f"{expand_prefix}{prompt}{expand_postfix}"
        think_template = cls._build_think_template(task, len(image_embeds))
        tokens = cls._tokenize_with_template(clip, think_prompt, think_template, image_embeds)

        generated = clip.generate(tokens, do_sample=False, max_length=max_new_tokens)
        generated_text = clip.decode(generated).strip()
        if len(generated_text) == 0:
            return prompt
        return f"{prompt} Here is a more detailed description. {generated_text}"

    @staticmethod
    def _parse_deepstack_layers(deepstack_layers: str):
        if deepstack_layers is None:
            return []
        values = []
        for item in deepstack_layers.split(","):
            item = item.strip()
            if len(item) == 0:
                continue
            values.append(int(item))
        return values

    @staticmethod
    def _encode_with_parity_options(clip, tokens, deepstack_layers, setclip):
        clip.cond_stage_model.reset_clip_options()
        clip.load_model(tokens)
        clip.cond_stage_model.set_clip_options(
            {
                "execution_device": clip.patcher.load_device,
                "deepstack": deepstack_layers,
                "setclip": setclip,
            }
        )
        encoded = clip.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = encoded[:2]
        pooled_dict = {"pooled_output": pooled}
        if len(encoded) > 2:
            pooled_dict.update(encoded[2])
        clip.add_hooks_to_dict(pooled_dict)
        return [[cond, pooled_dict]]

    @classmethod
    def execute(cls, clip, prompt, task, use_visual_inputs, max_visual_inputs, think, think_max_new_tokens, deepstack_layers, setclip, clip_vision_output=None) -> io.NodeOutput:
        cls._require_full_text_path(clip)
        cls._require_visual_inputs(task, use_visual_inputs, clip_vision_output)
        image_embeds = cls._extract_image_embeds(clip_vision_output, max_visual_inputs) if use_visual_inputs else []
        if think:
            prompt = cls._rewrite_prompt_with_think(clip, prompt, task, image_embeds, think_max_new_tokens)
        template = cls._build_template(task, len(image_embeds), add_generation_prompt=True)
        tokens = cls._tokenize_with_template(clip, prompt, template, image_embeds)
        deepstack = cls._parse_deepstack_layers(deepstack_layers)
        return io.NodeOutput(cls._encode_with_parity_options(clip, tokens, deepstack, setclip))


class HunyuanClipVisionOutputConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYOmniWeavingClipVisionConcat",
            display_name="HY OmniWeaving CLIP Vision Concat",
            category="conditioning/video_models",
            description="Concatenate multiple CLIP-Vision outputs for HY-OmniWeaving reference workflows.",
            inputs=[
                io.ClipVisionOutput.Input("clip_vision_output_1"),
                io.ClipVisionOutput.Input("clip_vision_output_2", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output_3", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output_4", optional=True),
            ],
            outputs=[
                io.ClipVisionOutput.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip_vision_output_1, clip_vision_output_2=None, clip_vision_output_3=None, clip_vision_output_4=None) -> io.NodeOutput:
        outputs = [o for o in (clip_vision_output_1, clip_vision_output_2, clip_vision_output_3, clip_vision_output_4) if o is not None]
        merged = comfy.clip_vision.Output()
        tensor_attrs = ["last_hidden_state", "image_embeds", "penultimate_hidden_states", "all_hidden_states", "mm_projected"]
        for attr in tensor_attrs:
            values = [getattr(o, attr) for o in outputs if hasattr(o, attr)]
            if len(values) > 0 and torch.is_tensor(values[0]):
                setattr(merged, attr, torch.cat(values, dim=0))

        image_sizes = []
        for o in outputs:
            if hasattr(o, "image_sizes"):
                image_sizes.extend(getattr(o, "image_sizes"))
        if len(image_sizes) > 0:
            merged.image_sizes = image_sizes
        return io.NodeOutput(merged)


class HunyuanVideo15OmniConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYOmniWeavingConditioning",
            display_name="HY OmniWeaving Conditioning",
            category="conditioning/video_models",
            description="HY-OmniWeaving-oriented i2v/t2v conditioning node with original-style Lanczos+center image preparation.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Combo.Input("task", options=["t2v", "i2v", "interpolation", "reference2v", "editing", "tiv2v"], default="t2v"),
                io.Int.Input("width", default=848, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("reference_images", optional=True),
                io.Image.Input("condition_video", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @staticmethod
    def _latent_length(length: int) -> int:
        return ((length - 1) // 4) + 1

    @staticmethod
    def _upscale_frames(frames: torch.Tensor, width: int, height: int):
        return comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)

    @classmethod
    def _encode_single_image(cls, vae, image: torch.Tensor, width: int, height: int):
        upscaled = cls._upscale_frames(image[:1], width, height)
        return vae.encode(upscaled[:, :, :, :3])

    @classmethod
    def _encode_video(cls, vae, video: torch.Tensor, width: int, height: int, length: int):
        upscaled = cls._upscale_frames(video[:length], width, height)
        return vae.encode(upscaled[:, :, :, :3])

    @staticmethod
    def _assign_frame(target: torch.Tensor, source: torch.Tensor, frame_idx: int):
        if frame_idx < 0 or frame_idx >= target.shape[2]:
            return
        target[:, :, frame_idx:frame_idx + 1] = source[:, :, :1]

    @classmethod
    def execute(cls, positive, negative, vae, task, width, height, length, batch_size, reference_images=None, condition_video=None, clip_vision_output=None) -> io.NodeOutput:
        latent_length = cls._latent_length(length)
        latent = torch.zeros([batch_size, 32, latent_length, height // 16, width // 16], device=comfy.model_management.intermediate_device())

        if task == "t2v":
            if clip_vision_output is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
            return io.NodeOutput(positive, negative, {"samples": latent})

        cond_latent = torch.zeros_like(latent[:1])
        omni_mask = torch.zeros((latent_length,), device=cond_latent.device, dtype=cond_latent.dtype)

        if task == "i2v":
            if reference_images is None or reference_images.shape[0] < 1:
                raise ValueError("Task i2v requires at least one reference image.")
            encoded = cls._encode_single_image(vae, reference_images, width, height)
            cls._assign_frame(cond_latent, encoded, 0)
            omni_mask[0] = 1.0
        elif task == "interpolation":
            if reference_images is None or reference_images.shape[0] < 2:
                raise ValueError("Task interpolation requires at least two reference images.")
            encoded_first = cls._encode_single_image(vae, reference_images[:1], width, height)
            encoded_last = cls._encode_single_image(vae, reference_images[-1:], width, height)
            cls._assign_frame(cond_latent, encoded_first, 0)
            cls._assign_frame(cond_latent, encoded_last, latent_length - 1)
            omni_mask[0] = 1.0
            omni_mask[-1] = 1.0
        elif task == "reference2v":
            if reference_images is None or reference_images.shape[0] < 1:
                raise ValueError("Task reference2v requires at least one reference image.")
            num_refs = min(reference_images.shape[0], max(1, latent_length - 1))
            for idx in range(num_refs):
                encoded = cls._encode_single_image(vae, reference_images[idx:idx + 1], width, height)
                frame_idx = min(idx + 1, latent_length - 1)
                cls._assign_frame(cond_latent, encoded, frame_idx)
                omni_mask[frame_idx] = 1.0
        elif task == "editing":
            if condition_video is None or condition_video.shape[0] < 1:
                raise ValueError("Task editing requires condition_video.")
            encoded = cls._encode_video(vae, condition_video, width, height, length)
            valid_frames = min(latent_length, encoded.shape[2])
            cond_latent[:, :, :valid_frames] = encoded[:, :, :valid_frames]
            omni_mask[:valid_frames] = 1.0
        elif task == "tiv2v":
            if condition_video is None or condition_video.shape[0] < 1:
                raise ValueError("Task tiv2v requires condition_video.")
            if reference_images is None or reference_images.shape[0] < 1:
                raise ValueError("Task tiv2v requires at least one reference image.")
            encoded_video = cls._encode_video(vae, condition_video, width, height, length)
            valid_frames = min(latent_length, encoded_video.shape[2])
            cond_latent[:, :, :valid_frames] = encoded_video[:, :, :valid_frames]
            omni_mask[:valid_frames] = 1.0
            encoded_ref = cls._encode_single_image(vae, reference_images[:1], width, height)
            ref_idx = 1 if latent_length > 1 else 0
            cond_latent[:, :, ref_idx:ref_idx + 1] += encoded_ref[:, :, :1]
            omni_mask[ref_idx] += 1.0

        cond_latent = comfy.utils.resize_to_batch_size(cond_latent, batch_size)
        concat_mask = (1.0 - omni_mask).view(1, 1, latent_length, 1, 1).expand(
            cond_latent.shape[0], 1, latent_length, cond_latent.shape[-2], cond_latent.shape[-1]
        ).to(cond_latent.dtype)

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": cond_latent, "concat_mask": concat_mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": cond_latent, "concat_mask": concat_mask})
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        return io.NodeOutput(positive, negative, {"samples": latent})


class HYOmniWeavingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HYOmniWeavingTextEncoderLoader,
            HYOmniWeavingUNetLoader,
            HYOmniWeavingVAELoader,
            TextEncodeHunyuanVideo15Omni,
            HunyuanClipVisionOutputConcat,
            HunyuanVideo15OmniConditioning,
        ]


async def comfy_entrypoint() -> HYOmniWeavingExtension:
    return HYOmniWeavingExtension()
