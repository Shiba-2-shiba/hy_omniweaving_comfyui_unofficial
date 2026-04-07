import types
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    torch.cuda.current_device = lambda: "cpu"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nodes


class _ClipStub:
    def __init__(self, has_byt5=True):
        self.cond_stage_model = types.SimpleNamespace(
            byt5_small=(object() if has_byt5 else None)
        )
        self.cond_stage_model.reset_clip_options = self._reset_clip_options
        self.cond_stage_model.set_clip_options = self._set_clip_options
        self.cond_stage_model.encode_token_weights = self._encode_token_weights
        self.patcher = types.SimpleNamespace(load_device="cpu")
        self.tokenize_calls = []
        self.generated = [1, 2, 3]
        self.clip_options = []

    def _reset_clip_options(self):
        self.clip_options.append(("reset", None))

    def _set_clip_options(self, options):
        self.clip_options.append(("set", options))

    def _encode_token_weights(self, tokens):
        self.last_encoded = tokens
        return ("cond_tensor", "pooled_tensor", {})

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        return {"tokens": text}

    def generate(self, tokens, do_sample=False, max_length=256):
        self.last_generate = {
            "tokens": tokens,
            "do_sample": do_sample,
            "max_length": max_length,
        }
        return self.generated

    def decode(self, token_ids):
        assert token_ids == self.generated
        return "expanded prompt"

    def load_model(self, tokens):
        self.loaded_tokens = tokens

    def add_hooks_to_dict(self, pooled_dict):
        self.last_pooled_dict = pooled_dict
        return pooled_dict


def test_hy_omniweaving_text_encode_requires_byt5_branch():
    clip = _ClipStub(has_byt5=False)

    with pytest.raises(ValueError, match="ByT5"):
        nodes.TextEncodeHunyuanVideo15Omni.execute(
            clip=clip,
            prompt="test",
            task="i2v",
            use_visual_inputs=False,
            max_visual_inputs=8,
            think=False,
            think_max_new_tokens=128,
            deepstack_layers="8,16,24",
            setclip=True,
            clip_vision_output=None,
        )


def test_hy_omniweaving_text_encode_requires_clip_vision_output_for_i2v_parity():
    clip = _ClipStub(has_byt5=True)

    with pytest.raises(ValueError, match="clip_vision_output"):
        nodes.TextEncodeHunyuanVideo15Omni.execute(
            clip=clip,
            prompt="test",
            task="i2v",
            use_visual_inputs=True,
            max_visual_inputs=8,
            think=False,
            think_max_new_tokens=128,
            deepstack_layers="8,16,24",
            setclip=True,
            clip_vision_output=None,
        )


def test_hy_omniweaving_text_encode_think_rewrites_prompt():
    clip = _ClipStub(has_byt5=True)

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="An anime girl dancing intensely",
        task="i2v",
        use_visual_inputs=False,
        max_visual_inputs=8,
        think=True,
        think_max_new_tokens=111,
        deepstack_layers="8,16,24",
        setclip=True,
        clip_vision_output=None,
    )

    assert len(clip.tokenize_calls) == 2
    assert "Please generate a more detailed description" in clip.tokenize_calls[0][0]
    assert clip.last_generate["do_sample"] is False
    assert clip.last_generate["max_length"] == 111
    assert "Here is a more detailed description. expanded prompt" in clip.tokenize_calls[1][0]
    assert "Here is a more detailed description. expanded prompt" in clip.last_encoded["tokens"]
    assert ("set", {"execution_device": "cpu", "deepstack": [8, 16, 24], "setclip": True}) in clip.clip_options


def test_hy_omniweaving_conditioning_uses_lanczos_center(monkeypatch):
    recorded = {}

    def fake_common_upscale(tensor, width, height, method, crop):
        recorded["width"] = width
        recorded["height"] = height
        recorded["method"] = method
        recorded["crop"] = crop
        return tensor

    monkeypatch.setattr(nodes.comfy.utils, "common_upscale", fake_common_upscale)

    frames = nodes.torch.zeros((1, 8, 8, 3))
    out = nodes.HunyuanVideo15OmniConditioning._upscale_frames(frames, 640, 640)

    assert tuple(out.shape) == (1, 8, 8, 3)
    assert recorded == {
        "width": 640,
        "height": 640,
        "method": "lanczos",
        "crop": "center",
    }


def test_convert_split_hy_omniweaving_attention_qkv_weight_and_bias():
    sd = {
        "double_blocks.0.img_attn_q.weight": torch.tensor([[1.0], [2.0]]),
        "double_blocks.0.img_attn_k.weight": torch.tensor([[3.0], [4.0]]),
        "double_blocks.0.img_attn_v.weight": torch.tensor([[5.0], [6.0]]),
        "double_blocks.0.img_attn_q.bias": torch.tensor([1.0, 2.0]),
        "double_blocks.0.img_attn_k.bias": torch.tensor([3.0, 4.0]),
        "double_blocks.0.img_attn_v.bias": torch.tensor([5.0, 6.0]),
    }

    converted_sd, converted, partial = nodes._convert_split_hy_omniweaving_attention_qkv(sd, strict_mode=True)

    assert converted == 2
    assert partial == []
    assert "double_blocks.0.img_attn.qkv.weight" in converted_sd
    assert "double_blocks.0.img_attn.qkv.bias" in converted_sd
    assert torch.equal(
        converted_sd["double_blocks.0.img_attn.qkv.weight"],
        torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
    )
    assert torch.equal(
        converted_sd["double_blocks.0.img_attn.qkv.bias"],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )


def test_convert_split_hy_omniweaving_attention_qkv_strict_mode_rejects_partial():
    sd = {
        "double_blocks.0.img_attn_q.weight": torch.tensor([[1.0]]),
        "double_blocks.0.img_attn_k.weight": torch.tensor([[2.0]]),
    }

    with pytest.raises(ValueError, match="Partial HY-OmniWeaving"):
        nodes._convert_split_hy_omniweaving_attention_qkv(sd, strict_mode=True)


def test_build_decoder_ddconfig_if_needed_returns_override_when_decoder_channels_differ():
    sd = {
        "decoder.conv_in.weight": torch.zeros((256, 4, 3, 3)),
    }
    ddconfig = {
        "ch": 128,
        "ch_mult": [1, 2, 4],
    }

    out = nodes._build_decoder_ddconfig_if_needed(sd, ddconfig)

    assert out is not None
    assert out["ch"] == 64
    assert ddconfig["ch"] == 128


def test_build_decoder_ddconfig_if_needed_returns_none_when_decoder_channels_match():
    sd = {
        "decoder.conv_in.weight": torch.zeros((512, 4, 3, 3)),
    }
    ddconfig = {
        "ch": 128,
        "ch_mult": [1, 2, 4],
    }

    out = nodes._build_decoder_ddconfig_if_needed(sd, ddconfig)

    assert out is None
