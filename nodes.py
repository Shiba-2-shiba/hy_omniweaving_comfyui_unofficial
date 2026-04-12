import json
import logging
import math
import node_helpers
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.clip_vision
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

try:
    from .runtime_patches import (
        ensure_runtime_patches,
        ensure_hy_omniweaving_deepstack_support,
        ensure_hy_omniweaving_text_encoder_support,
        extract_hy_omniweaving_mm_in_state_dict,
    )
except ImportError:
    from runtime_patches import (
        ensure_runtime_patches,
        ensure_hy_omniweaving_deepstack_support,
        ensure_hy_omniweaving_text_encoder_support,
        extract_hy_omniweaving_mm_in_state_dict,
    )


def _debug_enabled() -> bool:
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


def _clip_vision_shapes(clip_vision_output):
    if clip_vision_output is None:
        return {
            "present": False,
            "last_hidden_state": None,
            "penultimate_hidden_states": None,
            "image_embeds": None,
            "mm_projected": None,
        }
    return {
        "present": True,
        "last_hidden_state": _shape_of(getattr(clip_vision_output, "last_hidden_state", None)),
        "penultimate_hidden_states": _shape_of(getattr(clip_vision_output, "penultimate_hidden_states", None)),
        "image_embeds": _shape_of(getattr(clip_vision_output, "image_embeds", None)),
        "mm_projected": _shape_of(getattr(clip_vision_output, "mm_projected", None)),
    }


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
    ensure_runtime_patches()
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
    ensure_hy_omniweaving_text_encoder_support(clip)
    _debug_log(
        "text encoder prepared qwen_keys=%s byt5_keys=%s byt5_branch=%s patched=%s",
        len(qwen_sd),
        len(byt5_sd) if isinstance(byt5_sd, dict) else "n/a",
        _clip_has_byt5_branch(clip),
        hasattr(getattr(clip, "cond_stage_model", None), "_hy_omniweaving_text_encoder_patched"),
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


def _prepare_omniweaving_images(images: torch.Tensor, width: int, height: int):
    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)


class _ReduxImageEncoder(nn.Module):
    def __init__(self, redux_dim: int = 1152, txt_in_features: int = 4096):
        super().__init__()
        self.redux_up = nn.Linear(redux_dim, txt_in_features * 3)
        self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.redux_down(F.silu(self.redux_up(x)))


_HY_OMNIWEAVING_REDUX_VISION_CACHE = {}


def _preferred_clip_vision_model_name(*preferred_substrings: str) -> str | None:
    options = folder_paths.get_filename_list("clip_vision")
    if len(options) == 0:
        return None
    lowered_pairs = [(option, option.lower()) for option in options]
    for substring in preferred_substrings:
        lowered_substring = substring.lower()
        for option, lowered_option in lowered_pairs:
            if lowered_substring in lowered_option:
                return option
    return options[0]


def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bundled_redux_config_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "configs", filename)


def _resolve_redux_model_file(path: str, *, default_filenames: tuple[str, ...]) -> str:
    if not isinstance(path, str) or len(path.strip()) == 0:
        raise ValueError("Expected a non-empty model path.")

    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        full_model_path = folder_paths.get_full_path("clip_vision", path)
        if full_model_path is not None:
            candidates.append(full_model_path)
        candidates.append(os.path.join(os.path.dirname(__file__), path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        if os.path.isdir(candidate):
            for default_filename in default_filenames:
                default_path = os.path.join(candidate, default_filename)
                if os.path.isfile(default_path):
                    return os.path.abspath(default_path)

    raise FileNotFoundError(
        f"Could not find a compatible model file for '{path}'. "
        f"Expected a file directly or one of {default_filenames} inside the selected directory."
    )


def _infer_siglip_attention_heads(hidden_size: int) -> int:
    for preferred in (16, 12, 24, 18, 8):
        if hidden_size % preferred == 0:
            return preferred
    return 16 if hidden_size >= 1024 else 12


def _infer_siglip_vision_config_from_state_dict(sd: dict) -> dict:
    patch_embedding = sd.get("vision_model.embeddings.patch_embedding.weight")
    position_embedding = sd.get("vision_model.embeddings.position_embedding.weight")
    mlp_fc1 = sd.get("vision_model.encoder.layers.0.mlp.fc1.weight")
    layer_indices = []
    for key in sd.keys():
        if not key.startswith("vision_model.encoder.layers."):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        try:
            layer_indices.append(int(parts[3]))
        except ValueError:
            continue

    if not torch.is_tensor(patch_embedding) or not torch.is_tensor(position_embedding) or not torch.is_tensor(mlp_fc1):
        raise ValueError("Could not infer SigLIP config from state_dict; expected patch embedding, position embedding, and MLP tensors.")

    patch_size = int(patch_embedding.shape[-1])
    num_channels = int(patch_embedding.shape[1]) if patch_embedding.ndim >= 4 else 3
    hidden_size = int(mlp_fc1.shape[1])
    intermediate_size = int(mlp_fc1.shape[0])
    token_count = int(position_embedding.shape[0])
    side_tokens = int(round(math.sqrt(token_count)))
    if side_tokens * side_tokens != token_count:
        adjusted = int(round(math.sqrt(max(1, token_count - 1))))
        if adjusted * adjusted == token_count - 1:
            side_tokens = adjusted
        else:
            raise ValueError(f"Unsupported SigLIP position embedding length {token_count}; cannot infer image_size cleanly.")

    return {
        "architectures": ["SiglipVisionModel"],
        "attention_dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": hidden_size,
        "image_size": side_tokens * patch_size,
        "intermediate_size": intermediate_size,
        "layer_norm_eps": 1e-06,
        "model_type": "siglip_vision_model",
        "num_attention_heads": _infer_siglip_attention_heads(hidden_size),
        "num_channels": num_channels,
        "num_hidden_layers": (max(layer_indices) + 1) if len(layer_indices) > 0 else 0,
        "patch_size": patch_size,
    }


def _select_siglip_vision_config(sd: dict) -> dict:
    inferred = _infer_siglip_vision_config_from_state_dict(sd)
    bundled = _load_json_file(_bundled_redux_config_path("redux_image_encoder_config.json"))
    compare_keys = (
        "hidden_size",
        "image_size",
        "intermediate_size",
        "num_attention_heads",
        "num_channels",
        "num_hidden_layers",
        "patch_size",
    )
    if all(int(bundled.get(key, -1)) == int(inferred.get(key, -2)) for key in compare_keys):
        merged = inferred.copy()
        merged.update(bundled)
        return merged
    return inferred


def _infer_redux_embedder_config_from_state_dict(sd: dict) -> dict:
    redux_up = sd.get("redux_up.weight")
    redux_down = sd.get("redux_down.weight")
    if not torch.is_tensor(redux_up) or not torch.is_tensor(redux_down):
        raise ValueError("Could not infer Redux embedder config from state_dict; expected redux_up.weight and redux_down.weight.")

    return {
        "_class_name": "ReduxImageEncoder",
        "redux_dim": int(redux_up.shape[1]),
        "txt_in_features": int(redux_down.shape[0]),
    }


def _select_redux_embedder_config(sd: dict) -> dict:
    inferred = _infer_redux_embedder_config_from_state_dict(sd)
    bundled = _load_json_file(_bundled_redux_config_path("redux_image_embedder_config.json"))
    if (
        int(bundled.get("redux_dim", -1)) == int(inferred["redux_dim"])
        and int(bundled.get("txt_in_features", -1)) == int(inferred["txt_in_features"])
    ):
        merged = inferred.copy()
        merged.update(bundled)
        return merged
    return inferred


def _redux_target_device_and_dtype(device: str):
    if device == "cpu":
        return torch.device("cpu"), torch.float32

    target_device = comfy.model_management.text_encoder_device()
    target_dtype = comfy.model_management.text_encoder_dtype(target_device)
    if target_dtype is None:
        target_dtype = torch.float32
    return target_device, target_dtype


def _load_hy_omniweaving_redux_vision_models(
    image_encoder_dir: str,
    image_embedder_dir: str,
    device: str = "default",
):
    encoder_file = _resolve_redux_model_file(
        image_encoder_dir,
        default_filenames=("model.safetensors", "image_encoder.safetensors"),
    )
    embedder_file = _resolve_redux_model_file(
        image_embedder_dir,
        default_filenames=("diffusion_pytorch_model.safetensors", "image_embedder.safetensors", "model.safetensors"),
    )
    target_device, target_dtype = _redux_target_device_and_dtype(device)
    cache_key = (encoder_file, embedder_file, str(target_device), str(target_dtype))
    cached = _HY_OMNIWEAVING_REDUX_VISION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from transformers import SiglipVisionConfig, SiglipVisionModel

    encoder_sd = comfy.utils.load_torch_file(encoder_file, safe_load=True)
    encoder_config_dict = _select_siglip_vision_config(encoder_sd)
    encoder_config = SiglipVisionConfig(
        attention_dropout=encoder_config_dict.get("attention_dropout", 0.0),
        hidden_act=encoder_config_dict.get("hidden_act", "gelu_pytorch_tanh"),
        hidden_size=encoder_config_dict["hidden_size"],
        image_size=encoder_config_dict["image_size"],
        intermediate_size=encoder_config_dict["intermediate_size"],
        layer_norm_eps=encoder_config_dict.get("layer_norm_eps", 1e-06),
        num_attention_heads=encoder_config_dict["num_attention_heads"],
        num_channels=encoder_config_dict.get("num_channels", 3),
        num_hidden_layers=encoder_config_dict["num_hidden_layers"],
        patch_size=encoder_config_dict["patch_size"],
    )
    encoder = SiglipVisionModel(encoder_config)
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        raise ValueError(
            f"Failed to load Redux image encoder cleanly. missing={missing} unexpected={unexpected}"
        )
    encoder = encoder.eval()
    encoder.requires_grad_(False)
    encoder = encoder.to(device=target_device, dtype=target_dtype)

    embedder_sd = comfy.utils.load_torch_file(embedder_file, safe_load=True)
    embedder_config = _select_redux_embedder_config(embedder_sd)
    embedder = _ReduxImageEncoder(
        redux_dim=embedder_config.get("redux_dim", 1152),
        txt_in_features=embedder_config.get("txt_in_features", 4096),
    )
    missing, unexpected = embedder.load_state_dict(embedder_sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        raise ValueError(
            f"Failed to load Redux image embedder cleanly. missing={missing} unexpected={unexpected}"
        )
    embedder = embedder.eval()
    embedder.requires_grad_(False)
    embedder = embedder.to(device=target_device, dtype=target_dtype)

    bundle = {
        "encoder": encoder,
        "embedder": embedder,
        "image_size": int(encoder_config_dict.get("image_size", getattr(encoder.config, "image_size", 384))),
        "image_mean": encoder_config_dict.get("image_mean", [0.5, 0.5, 0.5]),
        "image_std": encoder_config_dict.get("image_std", [0.5, 0.5, 0.5]),
        "device": target_device,
        "dtype": target_dtype,
    }
    _HY_OMNIWEAVING_REDUX_VISION_CACHE[cache_key] = bundle
    return bundle


def _preprocess_redux_vision_images(images: torch.Tensor, image_size: int, image_mean, image_std, crop: bool) -> torch.Tensor:
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("HY OmniWeaving Redux Vision Encode expects IMAGE input with shape [B, H, W, C].")

    pixel_values = images[:, :, :, :3].movedim(-1, 1).float().clamp(0.0, 1.0)
    _, _, height, width = pixel_values.shape
    if crop:
        scale = max(image_size / max(1, height), image_size / max(1, width))
        resized_height = max(image_size, int(round(height * scale)))
        resized_width = max(image_size, int(round(width * scale)))
        pixel_values = F.interpolate(
            pixel_values,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
        )
        top = max(0, (resized_height - image_size) // 2)
        left = max(0, (resized_width - image_size) // 2)
        pixel_values = pixel_values[:, :, top:top + image_size, left:left + image_size]
    else:
        pixel_values = F.interpolate(
            pixel_values,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )

    mean = torch.tensor(image_mean, dtype=pixel_values.dtype, device=pixel_values.device).view(1, -1, 1, 1)
    std = torch.tensor(image_std, dtype=pixel_values.dtype, device=pixel_values.device).view(1, -1, 1, 1)
    return pixel_values.sub(mean).div(std)


def _encode_hy_omniweaving_redux_clip_vision_output(
    images: torch.Tensor,
    image_encoder_dir: str,
    image_embedder_dir: str,
    crop: str = "center",
    device: str = "default",
):
    bundle = _load_hy_omniweaving_redux_vision_models(
        image_encoder_dir=image_encoder_dir,
        image_embedder_dir=image_embedder_dir,
        device=device,
    )
    target_device = bundle["device"]
    target_dtype = bundle["dtype"]
    crop_center = crop == "center"
    pixel_values = _preprocess_redux_vision_images(
        images.to(device=target_device),
        image_size=bundle["image_size"],
        image_mean=bundle["image_mean"],
        image_std=bundle["image_std"],
        crop=crop_center,
    ).to(dtype=target_dtype)

    with torch.no_grad():
        vision_outputs = bundle["encoder"](pixel_values=pixel_values, output_hidden_states=True)
        last_hidden_state = vision_outputs.last_hidden_state
        hidden_states = tuple(vision_outputs.hidden_states or ())
        penultimate_hidden_states = hidden_states[-2] if len(hidden_states) >= 2 else last_hidden_state
        all_hidden_states = torch.stack(hidden_states, dim=1) if len(hidden_states) > 0 else last_hidden_state.unsqueeze(1)
        mm_projected = bundle["embedder"](last_hidden_state)

    output = comfy.clip_vision.Output()
    intermediate_device = comfy.model_management.intermediate_device()
    output.last_hidden_state = last_hidden_state.to(intermediate_device)
    output.penultimate_hidden_states = penultimate_hidden_states.to(intermediate_device)
    output.all_hidden_states = all_hidden_states.to(intermediate_device)
    output.image_embeds = last_hidden_state.to(intermediate_device)
    output.mm_projected = mm_projected.to(intermediate_device)
    output.image_sizes = [pixel_values.shape[1:]] * pixel_values.shape[0]
    _debug_log(
        "redux vision encode crop=%s image_size=%s last_hidden_state=%s mm_projected=%s",
        crop,
        bundle["image_size"],
        _shape_of(output.last_hidden_state),
        _shape_of(output.mm_projected),
    )
    return output


def _ensure_video_latent_dims(latent: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(latent):
        raise TypeError("Expected VAE latent tensor output.")
    if latent.ndim == 4:
        return latent.unsqueeze(2)
    if latent.ndim != 5:
        raise ValueError(f"Expected 4D or 5D latent tensor, got shape {tuple(latent.shape)}")
    return latent


def _unwrap_decoded_image_tensor(decoded) -> torch.Tensor:
    if isinstance(decoded, dict):
        decoded = decoded.get("samples", decoded.get("sample"))
    elif isinstance(decoded, (list, tuple)):
        decoded = decoded[0]

    if not torch.is_tensor(decoded):
        raise TypeError("Expected decoded image tensor output from VAE.")

    if decoded.ndim == 5:
        if decoded.shape[-1] in (1, 3, 4):
            decoded = decoded[:, 0]
        elif decoded.shape[1] in (1, 3, 4):
            decoded = decoded[:, :, 0].movedim(1, -1)
        else:
            raise ValueError(f"Unsupported decoded video tensor shape {tuple(decoded.shape)}")
    elif decoded.ndim == 4:
        if decoded.shape[-1] in (1, 3, 4):
            pass
        elif decoded.shape[1] in (1, 3, 4):
            decoded = decoded.movedim(1, -1)
        else:
            raise ValueError(f"Unsupported decoded image tensor shape {tuple(decoded.shape)}")
    else:
        raise ValueError(f"Unsupported decoded tensor rank {decoded.ndim}")

    return decoded[:, :, :, :3].contiguous()


def _derive_i2v_semantic_conditioning(vae, reference_images: torch.Tensor, width: int, height: int, latent_length: int):
    prepared = _prepare_omniweaving_images(reference_images[:1], width, height)[:, :, :, :3]
    first_latent = _ensure_video_latent_dims(vae.encode(prepared))
    semantic_images = _unwrap_decoded_image_tensor(vae.decode(first_latent[:, :, :1]))
    semantic_images = semantic_images.to(device=prepared.device, dtype=prepared.dtype)
    semantic_latent = _ensure_video_latent_dims(vae.encode(semantic_images))

    cond_latent = torch.zeros(
        (first_latent.shape[0], first_latent.shape[1], latent_length, first_latent.shape[-2], first_latent.shape[-1]),
        dtype=semantic_latent.dtype,
        device=semantic_latent.device,
    )
    cond_latent[:, :, 0:1] = semantic_latent[:, :, :1]
    return cond_latent, semantic_images


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
        mm_in_sd = extract_hy_omniweaving_mm_in_state_dict(sd)
        if len(mm_in_sd) > 0:
            linear_1_norm = _norm_of(mm_in_sd.get("linear_1.weight"))
            linear_2_norm = _norm_of(mm_in_sd.get("linear_2.weight"))
            _debug_log(
                "unet loader detected mm_in tensors count=%s source_linear1_shape=%s source_linear1_norm=%.6f source_linear2_shape=%s source_linear2_norm=%.6f",
                len(mm_in_sd),
                _shape_of(mm_in_sd.get("linear_1.weight")),
                linear_1_norm if linear_1_norm is not None else -1.0,
                _shape_of(mm_in_sd.get("linear_2.weight")),
                linear_2_norm if linear_2_norm is not None else -1.0,
            )
        if converted > 0:
            logging.info(f"HYOmniWeavingUNetLoader converted {converted} split attention tensors to qkv format.")
        if len(partial) > 0 and not strict_mode:
            logging.warning(f"HYOmniWeavingUNetLoader encountered partial split attention tensors: {partial}")
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        if model is None:
            raise RuntimeError(f"Failed to load HY-OmniWeaving diffusion model: {unet_name}")
        attached = ensure_hy_omniweaving_deepstack_support(model, mm_in_sd=mm_in_sd)
        if attached:
            logging.info("HYOmniWeavingUNetLoader attached mm_in for deepstack support without model-detection patching.")
            diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
            mm_in = getattr(diffusion_model, "mm_in", None)
            if mm_in is not None:
                linear_1 = getattr(mm_in, "linear_1", None)
                linear_2 = getattr(mm_in, "linear_2", None)
                _debug_log(
                    "mm_in attached class=%s linear1_shape=%s linear1_dtype=%s linear1_norm=%.6f linear2_norm=%.6f",
                    type(mm_in).__name__,
                    tuple(linear_1.weight.shape) if linear_1 is not None else None,
                    getattr(linear_1.weight, "dtype", None) if linear_1 is not None else None,
                    float(linear_1.weight.float().norm().item()) if linear_1 is not None else -1.0,
                    float(linear_2.weight.float().norm().item()) if linear_2 is not None else -1.0,
                )
        elif len(mm_in_sd) > 0:
            logging.warning("HYOmniWeavingUNetLoader found mm_in weights but failed to attach them after model load.")
        _debug_log(
            "unet loader converted_qkv=%s partial_qkv=%s mm_in_attached=%s wrapper_added=%s",
            converted,
            len(partial),
            attached,
            getattr(model, "_hy_omniweaving_diffusion_wrapper_added", False),
        )
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
        ensure_runtime_patches()
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        _debug_log(
            "vae loader omni_layout=%s decoder_conv_in_shape=%s",
            _is_omniweaving_vae_state_dict(sd),
            tuple(sd["decoder.conv_in.conv.weight"].shape) if "decoder.conv_in.conv.weight" in sd else tuple(sd["decoder.conv_in.weight"].shape) if "decoder.conv_in.weight" in sd else None,
        )
        vae = HYOmniWeavingVAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return io.NodeOutput(vae)


class TextEncodeHunyuanVideo15Omni(io.ComfyNode):
    THINK_MAX_EFFECTIVE_NEW_TOKENS = 256
    THINK_MAX_REWRITE_CHARS = 2048
    REWRITE_SUPPRESSED_TOKEN_IDS = (
        151644,
        151646, 151647, 151648, 151649, 151650, 151651,
        151652, 151653, 151654, 151655, 151656,
        151657, 151658, 151659, 151660, 151661, 151662, 151663, 151664, 151665, 151666,
        151667, 151668,
    )
    THINK_MODES = ("legacy_rewrite", "merge_hidden")
    THINK_KEEP_TOKENS_BY_TASK = {
        "i2v": 32,
        "t2v": 112,
        "interpolation": 96,
    }
    NEGATIVE_PROMPT_HINTS = (
        "low quality",
        "bad anatomy",
        "blur",
        "blurry",
        "artifacts",
        "watermark",
        "worst quality",
        "deformed",
        "jpeg artifacts",
        "text",
    )
    TASK_SPECS = {
        "t2v": {
            "prompt_mode": 1,
            "crop_start": 108,
            "system_prompt": "You are a helpful assistant. Describe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video.",
        },
        "i2v": {
            "prompt_mode": 2,
            "crop_start": 92,
            "system_prompt": "You are a helpful assistant. Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter the image to introduce motion and evolution over time. Generate a video using this image as the first frame that meets the user's requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency.",
        },
        "reference2v": {
            "prompt_mode": 3,
            "crop_start": 102,
            "system_prompt": "You are a helpful assistant. Given a text instruction and one or more input images, you need to explain how to extract and combine key information from the input images to construct a new image as the video's first frame, and then how the user's text instruction should alter the image to introduce motion and evolution over time. Generate a video that meets the user's requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency.",
        },
        "interpolation": {
            "prompt_mode": 4,
            "crop_start": 109,
            "system_prompt": "You are a helpful assistant. Given a text instruction, an image as the first frame of the video, and another image as the last frame of the video, you need to analyze the visual trajectory required to transition from the start to the end. Determine how the elements in the first frame must evolve, move, or transform to align with the last frame based on the text instruction. Generate a video that seamlessly connects these two frames, ensuring the motion and evolution between them fulfill the text description while maintaining temporal consistency",
        },
        "editing": {
            "prompt_mode": 5,
            "crop_start": 90,
            "system_prompt": "You are a helpful assistant. Given a text instruction and an input video, you need to analyze the visual content and temporal dynamics of the input video, and then explain how the user's text instruction should modify the video's visual style, objects, or scene composition. Generate an edited video that meets the user's requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence.",
        },
        "tiv2v": {
            "prompt_mode": 6,
            "crop_start": 104,
            "system_prompt": "You are a helpful assistant. Given a text instruction, a reference image and an input video, you need to analyze the visual content and temporal dynamics of the input video, alongside the scene or subject characteristics of the reference image. Explain how the user's text instruction directs the application of the reference image's visual attributes onto the input video. Generate an edited video that meets the user's requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence.",
        },
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
                io.Combo.Input("think_mode", options=list(cls.THINK_MODES), default="legacy_rewrite", advanced=True),
                io.Int.Input("think_keep_tokens", default=0, min=0, max=2048, advanced=True),
                io.String.Input("deepstack_layers", default="8,16,24", advanced=True),
                io.Boolean.Input("setclip", default=True, advanced=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("reference_images", optional=True),
                io.Image.Input("semantic_images", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @staticmethod
    def _task_spec(task: str) -> dict:
        return TextEncodeHunyuanVideo15Omni.TASK_SPECS.get(task, TextEncodeHunyuanVideo15Omni.TASK_SPECS["t2v"])

    @staticmethod
    def _task_system_prompt(task: str) -> str:
        return TextEncodeHunyuanVideo15Omni._task_spec(task)["system_prompt"]

    @staticmethod
    def _task_prompt_mode(task: str) -> int:
        return TextEncodeHunyuanVideo15Omni._task_spec(task)["prompt_mode"]

    @staticmethod
    def _task_crop_start(task: str) -> int:
        return TextEncodeHunyuanVideo15Omni._task_spec(task)["crop_start"]

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
    def _tokenize_with_template(clip, text, template, image_embeds, visual_images=None):
        try:
            if visual_images is not None:
                return clip.tokenize(text, llama_template=template, images=visual_images)
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
            shapes = _clip_vision_shapes(clip_vision_output)
            logging.warning(
                "HYOmniWeavingTextEncode received clip_vision_output without mm_projected. "
                "Text-side image embeds will be disabled. Shapes: last_hidden_state=%s penultimate_hidden_states=%s image_embeds=%s",
                shapes["last_hidden_state"],
                shapes["penultimate_hidden_states"],
                shapes["image_embeds"],
            )
            return []
        if mm_projected.ndim == 2:
            return [mm_projected]
        count = min(mm_projected.shape[0], max_visual_inputs)
        return [mm_projected[i] for i in range(count)]

    @staticmethod
    def _extract_visual_images(images, max_visual_inputs: int):
        if images is None:
            return []
        count = min(images.shape[0], max_visual_inputs)
        return [images[i:i + 1, :, :, :3] for i in range(count)]

    @classmethod
    def _resolve_visual_payload(cls, task, use_visual_inputs, max_visual_inputs, reference_images=None, semantic_images=None, clip_vision_output=None):
        visual_images = []
        image_embeds = []
        visual_source = "none"
        if use_visual_inputs:
            visual_inputs = semantic_images if semantic_images is not None else reference_images
            visual_images = cls._extract_visual_images(visual_inputs, max_visual_inputs)
            if len(visual_images) > 0:
                visual_source = "semantic_images" if semantic_images is not None else "reference_images"
            else:
                image_embeds = cls._extract_image_embeds(clip_vision_output, max_visual_inputs)
                if len(image_embeds) > 0:
                    visual_source = "clip_vision_output.mm_projected"
                elif task in ("i2v", "interpolation", "reference2v", "tiv2v"):
                    logging.warning(
                        "HYOmniWeavingTextEncode has no usable text-side visual inputs for task '%s'. "
                        "Connect HY OmniWeaving Image Prep output to reference_images for parity-sensitive runs.",
                        task,
                    )
        return {
            "visual_images": visual_images,
            "image_embeds": image_embeds,
            "visual_source": visual_source,
            "visual_input_count": len(visual_images) if len(visual_images) > 0 else len(image_embeds),
        }

    @staticmethod
    def _require_full_text_path(clip):
        if _clip_has_byt5_branch(clip):
            return
        raise ValueError(
            "HY-OmniWeaving requires the ByT5-enabled text path for parity. "
            "Use a loader route that preserves both Qwen2.5-VL and ByT5/Glyph conditioning."
        )

    @staticmethod
    def _require_visual_inputs(task: str, use_visual_inputs: bool, clip_vision_output, reference_images, semantic_images):
        if not use_visual_inputs:
            return
        if task not in ("i2v", "interpolation", "reference2v", "editing", "tiv2v"):
            return
        if semantic_images is not None and semantic_images.shape[0] > 0:
            return
        if reference_images is not None and reference_images.shape[0] > 0:
            return
        if clip_vision_output is not None:
            return
        raise ValueError(
            f"Task '{task}' with use_visual_inputs=True requires semantic_images, reference_images, or clip_vision_output for HY-OmniWeaving parity."
        )

    @classmethod
    def _rewrite_prompt_with_think(cls, clip, prompt, task, image_embeds, visual_images, max_new_tokens: int) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Think mode currently requires a single string prompt.")
        if task not in ("t2v", "i2v", "interpolation"):
            raise ValueError("Think mode is currently intended only for t2v, i2v, or interpolation tasks.")
        if max_new_tokens <= 0:
            return prompt
        effective_max_new_tokens = min(max_new_tokens, cls.THINK_MAX_EFFECTIVE_NEW_TOKENS)

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
        think_visual_images = cls._prepare_think_visual_images(visual_images)
        think_visual_count = len(think_visual_images) if len(think_visual_images) > 0 else len(image_embeds)
        think_template = cls._build_think_template(task, think_visual_count)
        tokens = cls._tokenize_with_template(
            clip,
            think_prompt,
            think_template,
            image_embeds,
            visual_images=think_visual_images if len(think_visual_images) > 0 else None,
        )

        generated = cls._generate_with_rewrite_suppression(clip, tokens, effective_max_new_tokens)
        generated_text = cls._decode_generated_text(clip, generated, tokens)
        _debug_log(
            "think rewrite task=%s original_chars=%s generated_chars=%s visual_inputs=%s",
            task,
            len(prompt),
            len(generated_text),
            think_visual_count,
        )
        _debug_log(
            "think rewrite generated_text task=%s text=%s",
            task,
            json.dumps(generated_text, ensure_ascii=False),
        )
        if len(generated_text) == 0:
            return prompt
        if len(generated_text) > cls.THINK_MAX_REWRITE_CHARS:
            logging.warning(
                "HYOmniWeavingTextEncode rejected runaway think rewrite for task '%s' because generated text was too long (%s chars).",
                task,
                len(generated_text),
            )
            return prompt
        rewritten = f"{prompt} Here is a more detailed description. {generated_text}"
        _debug_log(
            "think rewrite result task=%s rewritten_chars=%s",
            task,
            len(rewritten),
        )
        _debug_log(
            "think rewrite rewritten_text task=%s text=%s",
            task,
            json.dumps(rewritten, ensure_ascii=False),
        )
        return rewritten

    @staticmethod
    def _parse_deepstack_layers(deepstack_layers: str):
        if deepstack_layers is None:
            return []
        if isinstance(deepstack_layers, (list, tuple)):
            return [int(item) for item in deepstack_layers]
        values = []
        for item in deepstack_layers.split(","):
            item = item.strip()
            if len(item) == 0:
                continue
            values.append(int(item))
        return values

    @staticmethod
    def _resize_visual_for_think(image: torch.Tensor, max_side: int = 560):
        if image.ndim != 4:
            return image
        _, height, width, channels = image.shape
        if channels < 1 or max(height, width) <= max_side:
            return image[:, :, :, :3]

        scale = max_side / max(height, width)
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        resized = F.interpolate(
            image[:, :, :, :3].movedim(-1, 1),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)
        return resized

    @classmethod
    def _prepare_think_visual_images(cls, visual_images):
        return [cls._resize_visual_for_think(image) for image in visual_images]

    @staticmethod
    def _decode_generated_text(clip, generated, tokens) -> str:
        decode_input = generated
        if torch.is_tensor(generated) and generated.ndim >= 2:
            decode_input = generated[0]
        elif isinstance(generated, (list, tuple)) and len(generated) == 1:
            decode_input = generated[0]

        token_input_ids = tokens.get("input_ids") if isinstance(tokens, dict) else None
        token_attention_mask = tokens.get("attention_mask") if isinstance(tokens, dict) else None
        if (
            torch.is_tensor(token_input_ids)
            and torch.is_tensor(token_attention_mask)
            and torch.is_tensor(decode_input)
            and decode_input.ndim == 1
        ):
            prompt_length = int(token_attention_mask[0].sum().item())
            prompt_ids = token_input_ids[0, :prompt_length]
            if prompt_length > 0 and decode_input.shape[0] >= prompt_length and torch.equal(decode_input[:prompt_length], prompt_ids):
                decode_input = decode_input[prompt_length:]

        try:
            return clip.decode(decode_input, skip_special_tokens=True).strip()
        except TypeError:
            return clip.decode(decode_input).strip()

    @staticmethod
    def _rewrite_generation_target(clip):
        cond_stage_model = getattr(clip, "cond_stage_model", None)
        clip_name = getattr(cond_stage_model, "clip", "qwen25_7b")
        clip_model = getattr(cond_stage_model, clip_name, None)
        if clip_model is None:
            return None
        return getattr(clip_model, "transformer", clip_model)

    @classmethod
    def _generate_with_rewrite_suppression(cls, clip, tokens, max_new_tokens: int):
        target = cls._rewrite_generation_target(clip)
        previous = None
        had_previous = False
        if target is not None:
            had_previous = hasattr(target, "_hy_suppressed_token_ids")
            previous = getattr(target, "_hy_suppressed_token_ids", None)
            target._hy_suppressed_token_ids = cls.REWRITE_SUPPRESSED_TOKEN_IDS
            _debug_log(
                "rewrite suppression enabled token_count=%s first_ids=%s target=%s",
                len(cls.REWRITE_SUPPRESSED_TOKEN_IDS),
                cls.REWRITE_SUPPRESSED_TOKEN_IDS[:8],
                type(target).__name__,
            )
        try:
            return clip.generate(tokens, do_sample=False, max_length=max_new_tokens)
        finally:
            if target is not None:
                if had_previous:
                    target._hy_suppressed_token_ids = previous
                elif hasattr(target, "_hy_suppressed_token_ids"):
                    delattr(target, "_hy_suppressed_token_ids")

    @classmethod
    def _build_think_conditioning_prompt(cls, task: str, prompt: str) -> str:
        if task == "i2v":
            suffix = (
                "Starting from the provided first frame, preserve the same subject identity, background, "
                "layout, lighting, and overall framing. Make the temporal progression explicit with motion, "
                "pose change, expression change, and event order while keeping strong first-frame anchoring. "
                "Do not add new objects, new background elements, visible text, or a new camera setup."
            )
        elif task == "interpolation":
            suffix = (
                "Explain the transition between the boundary frames with explicit intermediate motion, "
                "camera movement, and temporal continuity, without adding new objects."
            )
        else:
            suffix = (
                "Make the temporal progression explicit while preserving the same subject, action, "
                "mood, count, and event order."
            )
        return f"{prompt.rstrip()}\n{suffix}"

    @classmethod
    def _is_negative_prompt_like(cls, prompt) -> bool:
        if not isinstance(prompt, str):
            return False
        normalized = prompt.strip().lower()
        if len(normalized) == 0:
            return False
        if normalized.startswith("negative prompt:"):
            return True
        hit_count = sum(1 for hint in cls.NEGATIVE_PROMPT_HINTS if hint in normalized)
        return hit_count >= 2

    @classmethod
    def _should_skip_think_merge(cls, prompt, task) -> bool:
        if not isinstance(prompt, str):
            return True
        if task not in cls.THINK_KEEP_TOKENS_BY_TASK:
            return True
        if len(prompt.strip()) == 0:
            return True
        return cls._is_negative_prompt_like(prompt)

    @staticmethod
    def _tail_tokens(value, keep_tokens: int, dim: int):
        if not torch.is_tensor(value):
            return value
        if keep_tokens <= 0 or value.shape[dim] <= keep_tokens:
            return value
        index = [slice(None)] * value.ndim
        index[dim] = slice(value.shape[dim] - keep_tokens, None)
        return value[tuple(index)].contiguous()

    @classmethod
    def _resolve_effective_keep_tokens(cls, task: str, think_keep_tokens: int, think_encoding: dict) -> int:
        requested = think_keep_tokens if think_keep_tokens > 0 else cls.THINK_KEEP_TOKENS_BY_TASK.get(task, 0)
        lengths = []
        cond = think_encoding.get("cond")
        extra = think_encoding.get("extra", {})
        if torch.is_tensor(cond) and cond.ndim >= 2:
            lengths.append(cond.shape[1])
        deepstack = extra.get("all_stack_text_states")
        if torch.is_tensor(deepstack) and deepstack.ndim >= 3:
            lengths.append(deepstack.shape[2])
        attention_mask = extra.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
            lengths.append(attention_mask.shape[1])
        if len(lengths) == 0:
            return max(requested, 0)
        if requested <= 0:
            return min(lengths)
        return min([requested, *lengths])

    @classmethod
    def _merge_encoded_conditioning(cls, base_encoding: dict, think_encoding: dict, task: str, think_keep_tokens: int):
        effective_keep_tokens = cls._resolve_effective_keep_tokens(task, think_keep_tokens, think_encoding)
        if effective_keep_tokens <= 0:
            return base_encoding

        base_cond = base_encoding.get("cond")
        think_cond = cls._tail_tokens(think_encoding.get("cond"), effective_keep_tokens, dim=1)
        if not torch.is_tensor(base_cond) or not torch.is_tensor(think_cond):
            return base_encoding

        merged_extra = dict(base_encoding.get("extra", {}))
        merged_cond = torch.cat([base_cond, think_cond], dim=1)

        base_deepstack = merged_extra.get("all_stack_text_states")
        think_deepstack = cls._tail_tokens(think_encoding.get("extra", {}).get("all_stack_text_states"), effective_keep_tokens, dim=2)
        if torch.is_tensor(base_deepstack) and torch.is_tensor(think_deepstack):
            merged_extra["all_stack_text_states"] = torch.cat([base_deepstack, think_deepstack], dim=2)

        base_attention_mask = merged_extra.get("attention_mask")
        think_attention_mask = cls._tail_tokens(think_encoding.get("extra", {}).get("attention_mask"), effective_keep_tokens, dim=1)
        if torch.is_tensor(base_attention_mask) and torch.is_tensor(think_attention_mask):
            merged_extra["attention_mask"] = torch.cat([base_attention_mask, think_attention_mask], dim=1)

        merged_extra["pooled_output"] = base_encoding["pooled_output"]
        _debug_log(
            "think merge task=%s keep_tokens=%s base_cond_shape=%s think_cond_shape=%s merged_cond_shape=%s "
            "base_deepstack_shape=%s think_deepstack_shape=%s merged_deepstack_shape=%s",
            task,
            effective_keep_tokens,
            _shape_of(base_cond),
            _shape_of(think_encoding.get("cond")),
            _shape_of(merged_cond),
            _shape_of(base_deepstack),
            _shape_of(think_encoding.get("extra", {}).get("all_stack_text_states")),
            _shape_of(merged_extra.get("all_stack_text_states")),
        )
        return {
            **base_encoding,
            "cond": merged_cond,
            "pooled_output": base_encoding["pooled_output"],
            "extra": merged_extra,
            "think_tokens_kept": effective_keep_tokens,
        }

    @classmethod
    def _conditioning_output(cls, encoded_components: dict):
        return [[encoded_components["cond"], encoded_components["extra"]]]

    @staticmethod
    def _encode_with_parity_options(clip, tokens, deepstack_layers, setclip, crop_start: int | None, task: str, visual_input_count: int = 0):
        clip.cond_stage_model.reset_clip_options()
        clip.load_model(tokens)
        clip.cond_stage_model.set_clip_options(
            {
                "execution_device": clip.patcher.load_device,
                "deepstack": deepstack_layers,
                "setclip": setclip,
                "crop_start": crop_start,
                "task_name": task,
                "visual_input_count": visual_input_count,
            }
        )
        encoded = clip.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = encoded[:2]
        pooled_dict = {"pooled_output": pooled}
        if len(encoded) > 2:
            pooled_dict.update(encoded[2])
        clip.add_hooks_to_dict(pooled_dict)
        _debug_log(
            "text encode output task=%s crop_start=%s visual_input_count=%s cond_shape=%s pooled_keys=%s all_stack_text_states_shape=%s attention_mask_shape=%s",
            task,
            crop_start,
            visual_input_count,
            _shape_of(cond),
            sorted(pooled_dict.keys()),
            _shape_of(pooled_dict.get("all_stack_text_states")),
            _shape_of(pooled_dict.get("attention_mask")),
        )
        return {
            "cond": cond,
            "pooled_output": pooled,
            "extra": pooled_dict,
        }

    @classmethod
    def _encode_prompt_components(cls, clip, prompt, task, deepstack_layers, setclip, image_embeds, visual_images, visual_source):
        visual_input_count = len(visual_images) if len(visual_images) > 0 else len(image_embeds)
        template = cls._build_template(task, visual_input_count, add_generation_prompt=True)
        tokens = cls._tokenize_with_template(
            clip,
            prompt,
            template,
            image_embeds,
            visual_images=visual_images if len(visual_images) > 0 else None,
        )
        crop_start = cls._task_crop_start(task)
        encoded = cls._encode_with_parity_options(
            clip,
            tokens,
            cls._parse_deepstack_layers(deepstack_layers),
            setclip,
            crop_start,
            task,
            visual_input_count,
        )
        encoded.update(
            {
                "tokens": tokens,
                "prompt": prompt,
                "crop_start": crop_start,
                "visual_source": visual_source,
                "visual_input_count": visual_input_count,
            }
        )
        return encoded

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        task,
        use_visual_inputs,
        max_visual_inputs,
        think,
        think_max_new_tokens,
        deepstack_layers,
        setclip,
        reference_images=None,
        semantic_images=None,
        clip_vision_output=None,
        think_mode="legacy_rewrite",
        think_keep_tokens=0,
    ) -> io.NodeOutput:
        ensure_runtime_patches()
        ensure_hy_omniweaving_text_encoder_support(clip)
        cls._require_full_text_path(clip)
        cls._require_visual_inputs(task, use_visual_inputs, clip_vision_output, reference_images, semantic_images)
        if think_mode not in cls.THINK_MODES:
            raise ValueError(f"Unsupported think_mode '{think_mode}'. Expected one of {cls.THINK_MODES}.")
        vision_shapes = _clip_vision_shapes(clip_vision_output)
        visual_payload = cls._resolve_visual_payload(
            task,
            use_visual_inputs,
            max_visual_inputs,
            reference_images=reference_images,
            semantic_images=semantic_images,
            clip_vision_output=clip_vision_output,
        )
        _debug_log(
            "text encode task=%s prompt_mode=%s crop_start=%s image_embeds=%s visual_images=%s visual_source=%s think=%s think_mode=%s "
            "think_keep_tokens=%s deepstack=%s setclip=%s clip_vision=%s",
            task,
            cls._task_prompt_mode(task),
            cls._task_crop_start(task),
            len(visual_payload["image_embeds"]),
            len(visual_payload["visual_images"]),
            visual_payload["visual_source"],
            think,
            think_mode,
            think_keep_tokens,
            deepstack_layers,
            setclip,
            vision_shapes,
        )
        if think and think_mode == "merge_hidden":
            base_encoding = cls._encode_prompt_components(
                clip,
                prompt,
                task,
                deepstack_layers,
                setclip,
                visual_payload["image_embeds"],
                visual_payload["visual_images"],
                visual_payload["visual_source"],
            )
            if cls._should_skip_think_merge(prompt, task):
                _debug_log("think merge skipped task=%s prompt_len=%s", task, len(prompt) if isinstance(prompt, str) else None)
                return io.NodeOutput(cls._conditioning_output(base_encoding))

            think_prompt = cls._build_think_conditioning_prompt(task, prompt)
            think_encoding = cls._encode_prompt_components(
                clip,
                think_prompt,
                task,
                deepstack_layers,
                setclip,
                visual_payload["image_embeds"],
                visual_payload["visual_images"],
                visual_payload["visual_source"],
            )
            merged_encoding = cls._merge_encoded_conditioning(base_encoding, think_encoding, task, think_keep_tokens)
            _debug_log(
                "think merge metadata task=%s visual_source=%s crop_start=%s setclip=%s base_tokens=%s think_tokens_raw=%s think_tokens_kept=%s",
                task,
                merged_encoding["visual_source"],
                merged_encoding["crop_start"],
                setclip,
                _shape_of(base_encoding["cond"]),
                _shape_of(think_encoding["cond"]),
                merged_encoding.get("think_tokens_kept"),
            )
            return io.NodeOutput(cls._conditioning_output(merged_encoding))

        if think:
            prompt = cls._rewrite_prompt_with_think(
                clip,
                prompt,
                task,
                visual_payload["image_embeds"],
                visual_payload["visual_images"],
                think_max_new_tokens,
            )
        encoded = cls._encode_prompt_components(
            clip,
            prompt,
            task,
            deepstack_layers,
            setclip,
            visual_payload["image_embeds"],
            visual_payload["visual_images"],
            visual_payload["visual_source"],
        )
        return io.NodeOutput(cls._conditioning_output(encoded))


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


class HYOmniWeavingReduxVisionEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        clip_vision_options = folder_paths.get_filename_list("clip_vision")
        return io.Schema(
            node_id="HYOmniWeavingReduxVisionEncode",
            display_name="HY OmniWeaving Redux Vision Encode",
            category="conditioning/video_models",
            description="Encode images with a local SigLIP image encoder plus Redux image embedder and emit a single CLIP_VISION_OUTPUT with mm_projected populated.",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "image_encoder_model",
                    options=clip_vision_options,
                    default=_preferred_clip_vision_model_name("image_encorder", "image_encoder", "siglip"),
                ),
                io.Combo.Input(
                    "image_embedder_model",
                    options=clip_vision_options,
                    default=_preferred_clip_vision_model_name("image_embedder", "redux"),
                ),
                io.Combo.Input("crop", options=["center", "none"], default="center", advanced=True),
                io.Combo.Input("device", options=["default", "cpu"], default="default", advanced=True),
            ],
            outputs=[
                io.ClipVisionOutput.Output(),
            ],
        )

    @classmethod
    def execute(cls, images, image_encoder_model, image_embedder_model, crop="center", device="default") -> io.NodeOutput:
        output = _encode_hy_omniweaving_redux_clip_vision_output(
            images=images,
            image_encoder_dir=image_encoder_model,
            image_embedder_dir=image_embedder_model,
            crop=crop,
            device=device,
        )
        return io.NodeOutput(output)


class HYOmniWeavingImagePrep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYOmniWeavingImagePrep",
            display_name="HY OmniWeaving Image Prep",
            category="conditioning/video_models",
            description="Prepare reference images for OmniWeaving-aligned i2v/reference workflows using the original-style Lanczos resize and center crop before VAE and CLIP-Vision encoding.",
            inputs=[
                io.Image.Input("reference_images"),
                io.Int.Input("width", default=848, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
            ],
            outputs=[
                io.Image.Output(display_name="prepared_images"),
            ],
        )

    @classmethod
    def execute(cls, reference_images, width, height) -> io.NodeOutput:
        prepared = _prepare_omniweaving_images(reference_images, width, height)
        return io.NodeOutput(prepared)


class HYOmniWeavingI2VSemanticImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYOmniWeavingI2VSemanticImages",
            display_name="HY OmniWeaving I2V Semantic Images",
            category="conditioning/video_models",
            description="Derive OmniWeaving-style i2v semantic images through the VAE roundtrip so the same semantic frame can feed text-side multimodal input and CLIP-Vision encoding.",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("reference_images"),
                io.Int.Input("width", default=848, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
            ],
            outputs=[
                io.Image.Output(display_name="semantic_images"),
            ],
        )

    @classmethod
    def execute(cls, vae, reference_images, width, height) -> io.NodeOutput:
        semantic_images = _derive_i2v_semantic_conditioning(vae, reference_images, width, height, latent_length=1)[1]
        return io.NodeOutput(semantic_images)


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
        return _prepare_omniweaving_images(frames, width, height)

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
        _debug_log(
            "conditioning task=%s width=%s height=%s length=%s batch=%s reference_images=%s condition_video=%s clip_vision=%s",
            task,
            width,
            height,
            length,
            batch_size,
            _shape_of(reference_images),
            _shape_of(condition_video),
            _clip_vision_shapes(clip_vision_output),
        )

        cond_latent = torch.zeros_like(latent[:1])
        omni_mask = torch.zeros((latent_length,), device=cond_latent.device, dtype=cond_latent.dtype)
        guiding_frame_index = None
        if task == "t2v":
            pass
        elif task == "i2v":
            if reference_images is None or reference_images.shape[0] < 1:
                raise ValueError("Task i2v requires at least one reference image.")
            cond_latent = _derive_i2v_semantic_conditioning(vae, reference_images, width, height, latent_length)[0]
            omni_mask[0] = 1.0
            guiding_frame_index = 0
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
        _debug_log(
            "conditioning output task=%s cond_latent_shape=%s cond_latent_norm=%s concat_mask_shape=%s concat_mask_sum=%s",
            task,
            _shape_of(cond_latent),
            _norm_of(cond_latent),
            _shape_of(concat_mask),
            float(concat_mask.sum().item()) if torch.is_tensor(concat_mask) else None,
        )
        cond_values = {"concat_latent_image": cond_latent, "concat_mask": concat_mask}
        if guiding_frame_index is not None:
            cond_values["guiding_frame_index"] = guiding_frame_index
        positive = node_helpers.conditioning_set_values(positive, cond_values)
        negative = node_helpers.conditioning_set_values(negative, cond_values)
        if clip_vision_output is not None and task != "t2v":
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
        elif clip_vision_output is not None:
            logging.warning(
                "HYOmniWeavingConditioning ignored clip_vision_output for task 't2v' to keep text-only generation isolated."
            )

        return io.NodeOutput(positive, negative, {"samples": latent})


class HYOmniWeavingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HYOmniWeavingTextEncoderLoader,
            HYOmniWeavingUNetLoader,
            HYOmniWeavingVAELoader,
            TextEncodeHunyuanVideo15Omni,
            HYOmniWeavingReduxVisionEncode,
            HYOmniWeavingImagePrep,
            HYOmniWeavingI2VSemanticImages,
            HunyuanClipVisionOutputConcat,
            HunyuanVideo15OmniConditioning,
        ]


async def comfy_entrypoint() -> HYOmniWeavingExtension:
    return HYOmniWeavingExtension()
