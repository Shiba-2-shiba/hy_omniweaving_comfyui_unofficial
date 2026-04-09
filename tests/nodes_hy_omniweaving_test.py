import types
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    torch.cuda.current_device = lambda: "cpu"


def _install_test_stubs():
    if "node_helpers" not in sys.modules:
        node_helpers = types.ModuleType("node_helpers")
        node_helpers.conditioning_set_values = lambda conditioning, values: (conditioning, values)
        sys.modules["node_helpers"] = node_helpers

    if "folder_paths" not in sys.modules:
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_filename_list = lambda kind: [
            "qwen_2.5_vl_7b.safetensors",
            "qwen_2.5_vl_7b_finetuned_model.safetensors",
            "byt5_small.safetensors",
        ]
        folder_paths.get_full_path_or_raise = lambda kind, name: f"C:/models/{kind}/{name}"
        folder_paths.get_folder_paths = lambda kind: [f"C:/models/{kind}"]
        sys.modules["folder_paths"] = folder_paths

    if "comfy" not in sys.modules:
        comfy = types.ModuleType("comfy")
        sys.modules["comfy"] = comfy

        clip_vision = types.ModuleType("comfy.clip_vision")
        clip_vision.Output = type("Output", (), {})
        sys.modules["comfy.clip_vision"] = clip_vision
        comfy.clip_vision = clip_vision

        model_management = types.ModuleType("comfy.model_management")
        model_management.is_amd = lambda: False
        model_management.dtype_size = lambda dtype: 1
        model_management.vae_device = lambda: "cpu"
        model_management.vae_offload_device = lambda: "cpu"
        model_management.vae_dtype = lambda device, dtypes: torch.float32
        model_management.intermediate_device = lambda: "cpu"
        model_management.archive_model_dtypes = lambda model: None
        sys.modules["comfy.model_management"] = model_management
        comfy.model_management = model_management

        model_patcher = types.ModuleType("comfy.model_patcher")

        class _Patcher:
            def __init__(self, *args, **kwargs):
                pass

            def is_dynamic(self):
                return False

        model_patcher.CoreModelPatcher = _Patcher
        model_patcher.ModelPatcher = _Patcher
        sys.modules["comfy.model_patcher"] = model_patcher
        comfy.model_patcher = model_patcher

        utils = types.ModuleType("comfy.utils")

        def _state_dict_prefix_replace(sd, replacements):
            out = {}
            for key, value in sd.items():
                new_key = key
                for old, new in replacements.items():
                    if key.startswith(old):
                        new_key = new + key[len(old):]
                        break
                out[new_key] = value
            return out

        utils.state_dict_prefix_replace = _state_dict_prefix_replace
        utils.common_upscale = lambda tensor, width, height, method, crop: tensor
        utils.resize_to_batch_size = lambda tensor, batch_size: tensor if tensor.shape[0] == batch_size else tensor.expand(batch_size, *tensor.shape[1:])
        utils.load_torch_file = lambda path, safe_load=True, return_metadata=False: {} if not return_metadata else ({}, {})
        sys.modules["comfy.utils"] = utils
        comfy.utils = utils

        sd = types.ModuleType("comfy.sd")

        class _VAE:
            def __init__(self, *args, **kwargs):
                pass

            def throw_exception_if_invalid(self):
                return None

            def model_size(self):
                return None

        class _ClipType:
            HUNYUAN_VIDEO_15 = "HUNYUAN_VIDEO_15"

        sd.VAE = _VAE
        sd.CLIPType = _ClipType
        sd.load_clip = lambda *args, **kwargs: None
        sd.load_text_encoder_state_dicts = lambda state_dicts, embedding_directory=None, clip_type=None, model_options=None, disable_dynamic=False: object()
        sd.load_diffusion_model_state_dict = lambda *args, **kwargs: object()
        sd.AutoencoderKL = type("AutoencoderKL", (), {})
        sd.AutoencodingEngine = type("AutoencodingEngine", (), {})
        sys.modules["comfy.sd"] = sd
        comfy.sd = sd

    if "comfy.patcher_extension" not in sys.modules:
        comfy = sys.modules["comfy"]
        patcher_extension = types.ModuleType("comfy.patcher_extension")
        patcher_extension.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
        sys.modules["comfy.patcher_extension"] = patcher_extension
        comfy.patcher_extension = patcher_extension

    if "comfy_api.latest" not in sys.modules:
        comfy_api = types.ModuleType("comfy_api")
        latest = types.ModuleType("comfy_api.latest")

        class _NodeBase:
            pass

        class _Schema:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _InputFactory:
            @staticmethod
            def Input(*args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        class _OutputFactory:
            @staticmethod
            def Output(*args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        io = types.SimpleNamespace(
            ComfyNode=_NodeBase,
            NodeOutput=lambda *args: args,
            Schema=_Schema,
            Combo=_InputFactory,
            Boolean=_InputFactory,
            Int=_InputFactory,
            String=_InputFactory,
            Clip=types.SimpleNamespace(Input=_InputFactory.Input, Output=_OutputFactory.Output),
            Conditioning=types.SimpleNamespace(Input=_InputFactory.Input, Output=_OutputFactory.Output),
            ClipVisionOutput=types.SimpleNamespace(Input=_InputFactory.Input, Output=_OutputFactory.Output),
            Vae=types.SimpleNamespace(Input=_InputFactory.Input, Output=_OutputFactory.Output),
            Model=types.SimpleNamespace(Output=_OutputFactory.Output),
            Image=types.SimpleNamespace(Input=_InputFactory.Input, Output=_OutputFactory.Output),
            Latent=types.SimpleNamespace(Output=_OutputFactory.Output),
        )

        latest.ComfyExtension = type("ComfyExtension", (), {})
        latest.io = io
        sys.modules["comfy_api"] = comfy_api
        sys.modules["comfy_api.latest"] = latest


_install_test_stubs()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nodes
import omniweaving_vae
import runtime_patches


class _ClipStub:
    def __init__(self, has_byt5=True):
        self.qwen_branch_clip_options = []
        self.cond_stage_model = types.SimpleNamespace(
            byt5_small=(object() if has_byt5 else None),
            clip="qwen25_7b",
        )
        self.cond_stage_model.qwen25_7b = types.SimpleNamespace(
            reset_clip_options=self._qwen_reset_clip_options,
            set_clip_options=self._qwen_set_clip_options,
            encode_token_weights=self._qwen_encode_token_weights,
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

    def _qwen_reset_clip_options(self):
        self.qwen_branch_clip_options.append(("reset", None))

    def _qwen_set_clip_options(self, options):
        self.qwen_branch_clip_options.append(("set", options))

    def _qwen_encode_token_weights(self, token_weight_pairs):
        qwen_out = torch.ones((1, 3, 6, 2))
        qwen_extra = {"attention_mask": torch.ones((1, 6))}
        return qwen_out, None, qwen_extra

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        return {"tokens": text, "qwen25_7b": [[(151644, 1.0), (151644, 1.0), (1, 1.0), (2, 1.0)]]}

    def generate(self, tokens, do_sample=False, max_length=256):
        self.last_generate = {
            "tokens": tokens,
            "do_sample": do_sample,
            "max_length": max_length,
        }
        return self.generated

    def decode(self, token_ids, **kwargs):
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


def test_hy_omniweaving_text_encode_allows_t2v_without_clip_vision_output():
    clip = _ClipStub(has_byt5=True)

    out = nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="A lighthouse in a storm",
        task="t2v",
        use_visual_inputs=True,
        max_visual_inputs=8,
        think=False,
        think_max_new_tokens=128,
        deepstack_layers="8,16,24",
        setclip=True,
        clip_vision_output=None,
    )

    assert out[0][0][0] == "cond_tensor"
    assert out[0][0][1]["pooled_output"] == "pooled_tensor"
    assert "all_stack_text_states" in out[0][0][1]
    assert "Describe the video by detailing the following aspects" in clip.tokenize_calls[0][1]["llama_template"]


def test_hy_omniweaving_task_specs_match_original_prompt_modes():
    assert nodes.TextEncodeHunyuanVideo15Omni._task_prompt_mode("t2v") == 1
    assert nodes.TextEncodeHunyuanVideo15Omni._task_prompt_mode("i2v") == 2
    assert nodes.TextEncodeHunyuanVideo15Omni._task_prompt_mode("reference2v") == 3
    assert nodes.TextEncodeHunyuanVideo15Omni._task_prompt_mode("interpolation") == 4
    assert nodes.TextEncodeHunyuanVideo15Omni._task_prompt_mode("editing") == 5
    assert nodes.TextEncodeHunyuanVideo15Omni._task_prompt_mode("tiv2v") == 6

    assert nodes.TextEncodeHunyuanVideo15Omni._task_crop_start("t2v") == 108
    assert nodes.TextEncodeHunyuanVideo15Omni._task_crop_start("i2v") == 92
    assert nodes.TextEncodeHunyuanVideo15Omni._task_crop_start("reference2v") == 102
    assert nodes.TextEncodeHunyuanVideo15Omni._task_crop_start("interpolation") == 109
    assert nodes.TextEncodeHunyuanVideo15Omni._task_crop_start("editing") == 90
    assert nodes.TextEncodeHunyuanVideo15Omni._task_crop_start("tiv2v") == 104


def test_hy_omniweaving_i2v_prompt_matches_original_wording():
    prompt = nodes.TextEncodeHunyuanVideo15Omni._task_system_prompt("i2v")
    assert prompt.startswith("You are a helpful assistant. Describe the key features of the input image")
    assert "then explain how the user's text instruction should alter the image" in prompt


def test_extract_image_embeds_warns_when_mm_projected_is_missing(caplog):
    output = types.SimpleNamespace(
        last_hidden_state=torch.zeros((1, 729, 1152)),
        penultimate_hidden_states=torch.zeros((1, 729, 1152)),
        image_embeds=torch.zeros((1, 1152)),
        mm_projected=None,
    )

    with caplog.at_level("WARNING"):
        embeds = nodes.TextEncodeHunyuanVideo15Omni._extract_image_embeds(output, 8)

    assert embeds == []
    assert "without mm_projected" in caplog.text
    assert "last_hidden_state=(1, 729, 1152)" in caplog.text


def test_extract_image_embeds_uses_mm_projected_when_present():
    output = types.SimpleNamespace(
        mm_projected=torch.zeros((2, 16, 4096)),
    )

    embeds = nodes.TextEncodeHunyuanVideo15Omni._extract_image_embeds(output, 1)

    assert len(embeds) == 1
    assert tuple(embeds[0].shape) == (16, 4096)


def test_hy_omniweaving_text_encode_prefers_semantic_images_over_reference_images_for_i2v():
    clip = _ClipStub(has_byt5=True)
    reference_images = torch.zeros((1, 640, 640, 3))
    semantic_images = torch.ones((1, 320, 320, 3))
    clip_vision_output = types.SimpleNamespace(
        last_hidden_state=torch.zeros((1, 729, 1152)),
        penultimate_hidden_states=torch.zeros((1, 729, 1152)),
        image_embeds=torch.zeros((1, 729, 1152)),
        mm_projected=None,
    )

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="A dancer starts moving",
        task="i2v",
        use_visual_inputs=True,
        max_visual_inputs=8,
        think=False,
        think_max_new_tokens=128,
        deepstack_layers="8,16,24",
        setclip=True,
        reference_images=reference_images,
        semantic_images=semantic_images,
        clip_vision_output=clip_vision_output,
    )

    assert "images" in clip.tokenize_calls[0][1]
    assert len(clip.tokenize_calls[0][1]["images"]) == 1
    assert tuple(clip.tokenize_calls[0][1]["images"][0].shape) == (1, 320, 320, 3)


def test_hy_omniweaving_text_encode_warns_when_i2v_has_no_usable_text_side_visual_input(caplog):
    clip = _ClipStub(has_byt5=True)
    clip_vision_output = types.SimpleNamespace(
        last_hidden_state=torch.zeros((1, 729, 1152)),
        penultimate_hidden_states=torch.zeros((1, 729, 1152)),
        image_embeds=torch.zeros((1, 729, 1152)),
        mm_projected=None,
    )

    with caplog.at_level("WARNING"):
        nodes.TextEncodeHunyuanVideo15Omni.execute(
            clip=clip,
            prompt="A dancer starts moving",
            task="i2v",
            use_visual_inputs=True,
            max_visual_inputs=8,
            think=False,
            think_max_new_tokens=128,
            deepstack_layers="8,16,24",
            setclip=True,
            reference_images=None,
            semantic_images=None,
            clip_vision_output=clip_vision_output,
        )

    assert "no usable text-side visual inputs" in caplog.text


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
        semantic_images=None,
        clip_vision_output=None,
    )

    assert len(clip.tokenize_calls) == 2
    assert "Please generate a more detailed description" in clip.tokenize_calls[0][0]
    assert "Describe the key features of the input image" in clip.tokenize_calls[0][1]["llama_template"]
    assert clip.last_generate["do_sample"] is False
    assert clip.last_generate["max_length"] == 111
    assert "Here is a more detailed description. expanded prompt" in clip.tokenize_calls[1][0]
    assert "Here is a more detailed description. expanded prompt" in clip.last_encoded["tokens"]
    assert "Describe the key features of the input image" in clip.tokenize_calls[1][1]["llama_template"]
    assert (
        "set",
        {
            "execution_device": "cpu",
            "deepstack": [8, 16, 24],
            "setclip": True,
            "crop_start": 92,
            "task_name": "i2v",
            "visual_input_count": 0,
        },
    ) in clip.clip_options


def test_hy_omniweaving_text_encode_think_rejects_runaway_rewrite(monkeypatch):
    clip = _ClipStub(has_byt5=True)
    monkeypatch.setattr(
        nodes.TextEncodeHunyuanVideo15Omni,
        "_decode_generated_text",
        staticmethod(lambda clip, generated, tokens: "x" * 3000),
    )

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="An anime girl dancing intensely",
        task="t2v",
        use_visual_inputs=False,
        max_visual_inputs=8,
        think=True,
        think_max_new_tokens=1000,
        deepstack_layers="8,16,24",
        setclip=True,
        semantic_images=None,
        clip_vision_output=None,
    )

    assert len(clip.tokenize_calls) == 2
    assert clip.tokenize_calls[1][0] == "An anime girl dancing intensely"
    assert clip.last_generate["max_length"] == 256


def test_hy_omniweaving_text_encode_think_resizes_visual_inputs_for_ar_prompt():
    clip = _ClipStub(has_byt5=True)
    semantic_images = torch.zeros((1, 1024, 512, 3))

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="A dancer starts moving",
        task="i2v",
        use_visual_inputs=True,
        max_visual_inputs=8,
        think=True,
        think_max_new_tokens=128,
        deepstack_layers="8,16,24",
        setclip=True,
        reference_images=None,
        semantic_images=semantic_images,
        clip_vision_output=None,
    )

    think_images = clip.tokenize_calls[0][1]["images"]
    assert len(think_images) == 1
    assert tuple(think_images[0].shape[-3:-1]) == (560, 280)


def test_decode_generated_text_trims_prompt_prefix_only_when_it_matches():
    clip = _ClipStub(has_byt5=True)
    generated = torch.tensor([[101, 102, 201, 202]])
    tokens = {
        "input_ids": torch.tensor([[101, 102, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
    }

    seen = {}

    def decode(token_ids, **kwargs):
        seen["token_ids"] = token_ids
        seen["kwargs"] = kwargs
        return "tail only"

    clip.decode = decode

    out = nodes.TextEncodeHunyuanVideo15Omni._decode_generated_text(clip, generated, tokens)

    assert out == "tail only"
    assert torch.equal(seen["token_ids"], torch.tensor([201, 202]))
    assert seen["kwargs"] == {"skip_special_tokens": True}


def test_decode_generated_text_keeps_continuation_when_prefix_does_not_match():
    clip = _ClipStub(has_byt5=True)
    generated = torch.tensor([[301, 302, 201, 202]])
    tokens = {
        "input_ids": torch.tensor([[101, 102, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
    }

    seen = {}

    def decode(token_ids, **kwargs):
        seen["token_ids"] = token_ids
        seen["kwargs"] = kwargs
        return "full continuation"

    clip.decode = decode

    out = nodes.TextEncodeHunyuanVideo15Omni._decode_generated_text(clip, generated, tokens)

    assert out == "full continuation"
    assert torch.equal(seen["token_ids"], torch.tensor([301, 302, 201, 202]))
    assert seen["kwargs"] == {"skip_special_tokens": True}


def test_hy_omniweaving_text_encode_passes_visual_input_metadata_into_clip_options():
    clip = _ClipStub(has_byt5=True)
    semantic_images = torch.zeros((2, 320, 320, 3))

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="A dancer starts moving",
        task="i2v",
        use_visual_inputs=True,
        max_visual_inputs=1,
        think=False,
        think_max_new_tokens=128,
        deepstack_layers="8,16,24",
        setclip=True,
        reference_images=None,
        semantic_images=semantic_images,
        clip_vision_output=None,
    )

    assert (
        "set",
        {
            "execution_device": "cpu",
            "deepstack": [8, 16, 24],
            "setclip": True,
            "crop_start": 92,
            "task_name": "i2v",
            "visual_input_count": 1,
        },
    ) in clip.clip_options


def test_hy_omniweaving_text_encode_uses_reference_style_task_template():
    clip = _ClipStub(has_byt5=True)

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="test",
        task="reference2v",
        use_visual_inputs=False,
        max_visual_inputs=8,
        think=False,
        think_max_new_tokens=1000,
        deepstack_layers="8,16,24",
        setclip=True,
        semantic_images=None,
        clip_vision_output=None,
    )

    assert "Given a text instruction and one or more input images" in clip.tokenize_calls[0][1]["llama_template"]
    assert clip.tokenize_calls[0][1]["llama_template"].endswith("<|im_start|>assistant\n")


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


def test_hy_omniweaving_image_prep_uses_lanczos_center(monkeypatch):
    recorded = {}

    def fake_common_upscale(tensor, width, height, method, crop):
        recorded["shape"] = tuple(tensor.shape)
        recorded["width"] = width
        recorded["height"] = height
        recorded["method"] = method
        recorded["crop"] = crop
        return tensor

    monkeypatch.setattr(nodes.comfy.utils, "common_upscale", fake_common_upscale)

    images = nodes.torch.zeros((2, 8, 8, 3))
    out = nodes.HYOmniWeavingImagePrep.execute(reference_images=images, width=832, height=480)

    assert tuple(out[0].shape) == (2, 8, 8, 3)
    assert recorded == {
        "shape": (2, 3, 8, 8),
        "width": 832,
        "height": 480,
        "method": "lanczos",
        "crop": "center",
    }


def test_hy_omniweaving_i2v_semantic_images_roundtrip_reference_frame(monkeypatch):
    class _VAE:
        def __init__(self):
            self.encode_inputs = []
            self.decode_inputs = []

        def encode(self, image):
            self.encode_inputs.append(image.clone())
            return torch.full((1, 32, 1, 2, 2), float(len(self.encode_inputs)), dtype=image.dtype)

        def decode(self, latent):
            self.decode_inputs.append(latent.clone())
            return torch.full((1, 8, 8, 3), 0.75, dtype=latent.dtype)

    monkeypatch.setattr(nodes.comfy.utils, "common_upscale", lambda tensor, width, height, method, crop: tensor)

    semantic_images, = nodes.HYOmniWeavingI2VSemanticImages.execute(
        vae=_VAE(),
        reference_images=torch.zeros((1, 8, 8, 3)),
        width=832,
        height=480,
    )

    assert tuple(semantic_images.shape) == (1, 8, 8, 3)
    assert torch.equal(semantic_images, torch.full((1, 8, 8, 3), 0.75))


def test_hy_omniweaving_conditioning_i2v_sets_stock_comfy_mask_polarity():
    class _VAE:
        def __init__(self):
            self.encode_inputs = []
            self.decode_inputs = []

        def encode(self, image):
            self.encode_inputs.append(image.clone())
            return torch.full((1, 32, 1, 2, 2), float(len(self.encode_inputs)), dtype=image.dtype)

        def decode(self, latent):
            self.decode_inputs.append(latent.clone())
            return torch.full((1, 8, 8, 3), 0.5, dtype=latent.dtype)

    vae = _VAE()

    positive, negative, latent = nodes.HunyuanVideo15OmniConditioning.execute(
        positive="pos",
        negative="neg",
        vae=vae,
        task="i2v",
        width=32,
        height=32,
        length=5,
        batch_size=1,
        reference_images=torch.zeros((1, 8, 8, 3)),
        condition_video=None,
        clip_vision_output=None,
    )

    pos_values = positive[1]
    neg_values = negative[1]
    assert tuple(latent["samples"].shape) == (1, 32, 2, 2, 2)
    assert tuple(pos_values["concat_latent_image"].shape) == (1, 32, 2, 2, 2)
    assert tuple(pos_values["concat_mask"].shape) == (1, 1, 2, 2, 2)
    assert torch.equal(pos_values["concat_latent_image"][:, :, 0], torch.full((1, 32, 2, 2), 2.0))
    assert torch.equal(pos_values["concat_latent_image"][:, :, 1], torch.zeros((1, 32, 2, 2)))
    assert torch.equal(pos_values["concat_mask"][:, :, 0], torch.zeros((1, 1, 2, 2)))
    assert torch.equal(pos_values["concat_mask"][:, :, 1], torch.ones((1, 1, 2, 2)))
    assert torch.equal(neg_values["concat_mask"], pos_values["concat_mask"])
    assert pos_values["guiding_frame_index"] == 0
    assert tuple(pos_values["ref_latent"].shape) == (1, 32, 1, 2, 2)
    assert torch.equal(neg_values["ref_latent"], pos_values["ref_latent"])
    assert len(vae.encode_inputs) == 2
    assert len(vae.decode_inputs) == 1


def test_hy_omniweaving_conditioning_t2v_sets_zero_concat_latent_and_mask():
    positive, negative, latent = nodes.HunyuanVideo15OmniConditioning.execute(
        positive="pos",
        negative="neg",
        vae=object(),
        task="t2v",
        width=32,
        height=32,
        length=5,
        batch_size=2,
        reference_images=None,
        condition_video=None,
        clip_vision_output=None,
    )

    pos_values = positive[1]
    neg_values = negative[1]
    assert tuple(latent["samples"].shape) == (2, 32, 2, 2, 2)
    assert tuple(pos_values["concat_latent_image"].shape) == (2, 32, 2, 2, 2)
    assert tuple(pos_values["concat_mask"].shape) == (2, 1, 2, 2, 2)
    assert torch.equal(pos_values["concat_latent_image"], torch.zeros((2, 32, 2, 2, 2)))
    assert torch.equal(pos_values["concat_mask"], torch.ones((2, 1, 2, 2, 2)))
    assert torch.equal(neg_values["concat_latent_image"], pos_values["concat_latent_image"])
    assert torch.equal(neg_values["concat_mask"], pos_values["concat_mask"])
    assert "guiding_frame_index" not in pos_values
    assert "ref_latent" not in pos_values


def test_hy_omniweaving_conditioning_t2v_does_not_forward_clip_vision_output():
    clip_vision_output = types.SimpleNamespace(
        last_hidden_state=torch.zeros((1, 729, 1152)),
        penultimate_hidden_states=torch.zeros((1, 729, 1152)),
        image_embeds=torch.zeros((1, 729, 1152)),
        mm_projected=None,
    )

    positive, negative, _ = nodes.HunyuanVideo15OmniConditioning.execute(
        positive="pos",
        negative="neg",
        vae=object(),
        task="t2v",
        width=32,
        height=32,
        length=5,
        batch_size=1,
        reference_images=None,
        condition_video=None,
        clip_vision_output=clip_vision_output,
    )

    assert "clip_vision_output" not in positive[1]
    assert "clip_vision_output" not in negative[1]


def test_hy_omniweaving_conditioning_i2v_keeps_clip_vision_output():
    class _VAE:
        def encode(self, image):
            return torch.full((1, 32, 1, 2, 2), 1.0, dtype=image.dtype)

        def decode(self, latent):
            return torch.full((1, 8, 8, 3), 0.5, dtype=latent.dtype)

    clip_vision_output = types.SimpleNamespace(mm_projected=torch.zeros((1, 16, 4096)))

    positive, negative, _ = nodes.HunyuanVideo15OmniConditioning.execute(
        positive="pos",
        negative="neg",
        vae=_VAE(),
        task="i2v",
        width=32,
        height=32,
        length=5,
        batch_size=1,
        reference_images=torch.zeros((1, 8, 8, 3)),
        condition_video=None,
        clip_vision_output=clip_vision_output,
    )

    assert positive[1]["clip_vision_output"] is clip_vision_output
    assert negative[1]["clip_vision_output"] is clip_vision_output


def test_ensure_runtime_patches_is_idempotent(monkeypatch):
    calls = []

    monkeypatch.setattr(runtime_patches, "_patch_qwen25_think_generation", lambda: calls.append("think"))
    monkeypatch.setattr(runtime_patches, "_patch_autoencoder_legacy", lambda: calls.append("vae"))
    if hasattr(runtime_patches, "_HY_OMNIWEAVING_RUNTIME_PATCHES_READY"):
        monkeypatch.setattr(runtime_patches, "_HY_OMNIWEAVING_RUNTIME_PATCHES_READY", False)

    assert runtime_patches.ensure_runtime_patches() is True
    assert runtime_patches.ensure_runtime_patches() is False
    assert calls == ["think", "vae"]


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


def test_filter_known_optional_vae_missing_keys_removes_hunyuan_temb_proj_noise():
    missing = [
        "encoder.mid.block_1.temb_proj.weight",
        "decoder.mid.block_2.temb_proj.bias",
        "quant_conv.weight",
    ]

    filtered, ignored = nodes._filter_known_optional_vae_missing_keys(missing)

    assert filtered == ["quant_conv.weight"]
    assert ignored == [
        "encoder.mid.block_1.temb_proj.weight",
        "decoder.mid.block_2.temb_proj.bias",
    ]


def test_is_omniweaving_vae_state_dict_detects_causal_conv_layout():
    assert nodes._is_omniweaving_vae_state_dict(
        {
            "encoder.conv_in.conv.weight": torch.tensor([1.0]),
            "decoder.conv_in.conv.weight": torch.tensor([2.0]),
        }
    )
    assert not nodes._is_omniweaving_vae_state_dict(
        {
            "encoder.conv_in.weight": torch.tensor([1.0]),
            "decoder.conv_in.weight": torch.tensor([2.0]),
        }
    )


def test_load_omniweaving_vae_config_matches_reference_shape():
    config = nodes._load_omniweaving_vae_config()

    assert config["_class_name"] == "AutoencoderKLConv3D"
    assert config["latent_channels"] == 32
    assert config["ffactor_spatial"] == 16
    assert config["ffactor_temporal"] == 4


def test_hy_omniweaving_vae_uses_custom_omniweaving_loader_for_causal_conv_state_dict(monkeypatch):
    called = {}

    def fake_init(self, sd=None, device=None, dtype=None):
        called["sd"] = sd
        called["device"] = device
        called["dtype"] = dtype

    monkeypatch.setattr(nodes.HYOmniWeavingVAE, "_init_omniweaving_vae", fake_init)

    nodes.HYOmniWeavingVAE(
        sd={
            "encoder.conv_in.conv.weight": torch.zeros((128, 3, 3, 3, 3)),
            "decoder.conv_in.conv.weight": torch.zeros((1024, 32, 3, 3, 3)),
        },
        device="cpu",
        dtype=torch.float16,
    )

    assert called["device"] == "cpu"
    assert called["dtype"] == torch.float16


def test_omniweaving_decoder_adds_latent_residual_before_mid_blocks(monkeypatch):
    decoder = omniweaving_vae.Decoder(
        z_channels=2,
        out_channels=3,
        block_out_channels=[4],
        num_res_blocks=0,
        ffactor_spatial=1,
        ffactor_temporal=1,
    )

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x.repeat_interleave(2, dim=1) + 1

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    class Stage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.ModuleList([Identity()])

    decoder.conv_in = AddOne()
    decoder.mid.block_1 = Identity()
    decoder.mid.attn_1 = Identity()
    decoder.mid.block_2 = Identity()
    decoder.norm_out = Identity()
    decoder.conv_out = Identity()
    decoder.up = torch.nn.ModuleList([Stage()])

    z = torch.ones((1, 2, 1, 1, 1))
    out = decoder(z)

    expected = torch.full((1, 4, 1, 1, 1), torch.nn.functional.silu(torch.tensor(3.0)).item())
    assert torch.equal(out, expected)


def test_normalize_hy_omniweaving_text_encoder_state_dict_rewrites_reference_prefixes():
    sd = {
        "__metadata__": {"format": "pt"},
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.tensor([1.0]),
        "model.visual.blocks.0.attn.qkv.weight": torch.tensor([2.0]),
        "final_layer_norm.weight": torch.tensor([3.0]),
    }

    normalized = nodes._normalize_hy_omniweaving_text_encoder_state_dict(sd)

    assert "__metadata__" not in normalized
    assert "model.layers.0.self_attn.k_proj.weight" in normalized
    assert "visual.blocks.0.attn.qkv.weight" in normalized
    assert "model.norm.weight" in normalized


def test_load_hy_omniweaving_dual_text_encoder_normalizes_qwen_and_keeps_byt5(monkeypatch):
    loaded = {}

    def fake_get_full_path_or_raise(kind, name):
        return f"C:/models/{kind}/{name}"

    def fake_get_folder_paths(kind):
        return [f"C:/models/{kind}"]

    def fake_load_torch_file(path, safe_load=True):
        if path.endswith("qwen_2.5_vl_7b_finetuned_model.safetensors"):
            return {
                "model.language_model.layers.0.self_attn.k_proj.weight": torch.tensor([1.0]),
                "final_layer_norm.weight": torch.tensor([2.0]),
            }
        if path.endswith("byt5_small.safetensors"):
            return {
                "encoder.block.0.layer.0.SelfAttention.o.weight": torch.tensor([3.0]),
            }
        raise AssertionError(path)

    def fake_load_text_encoder_state_dicts(state_dicts, embedding_directory=None, clip_type=None, model_options=None, disable_dynamic=False):
        loaded["state_dicts"] = state_dicts
        loaded["embedding_directory"] = embedding_directory
        loaded["clip_type"] = clip_type
        loaded["model_options"] = model_options
        return "dual-clip"

    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", fake_get_full_path_or_raise)
    monkeypatch.setattr(nodes.folder_paths, "get_folder_paths", fake_get_folder_paths)
    monkeypatch.setattr(nodes.comfy.utils, "load_torch_file", fake_load_torch_file)
    monkeypatch.setattr(nodes.comfy.sd, "load_text_encoder_state_dicts", fake_load_text_encoder_state_dicts)

    clip = nodes._load_hy_omniweaving_dual_text_encoder(
        qwen_text_encoder="qwen_2.5_vl_7b_finetuned_model.safetensors",
        byt5_text_encoder="byt5_small.safetensors",
        device="cpu",
    )

    assert clip == "dual-clip"
    assert loaded["embedding_directory"] == ["C:/models/embeddings"]
    assert loaded["clip_type"] == nodes.comfy.sd.CLIPType.HUNYUAN_VIDEO_15
    assert loaded["model_options"]["load_device"] == torch.device("cpu")
    assert loaded["model_options"]["offload_device"] == torch.device("cpu")
    assert "model.layers.0.self_attn.k_proj.weight" in loaded["state_dicts"][0]
    assert "model.norm.weight" in loaded["state_dicts"][0]
    assert "encoder.block.0.layer.0.SelfAttention.o.weight" in loaded["state_dicts"][1]


def test_hy_omniweaving_text_encoder_loader_outputs_dual_clip(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_load_hy_omniweaving_dual_text_encoder",
        lambda qwen_text_encoder, byt5_text_encoder, device="default": {
            "qwen": qwen_text_encoder,
            "byt5": byt5_text_encoder,
            "device": device,
        },
    )

    out = nodes.HYOmniWeavingTextEncoderLoader.execute(
        qwen_text_encoder="qwen_2.5_vl_7b_finetuned_model.safetensors",
        byt5_text_encoder="byt5_small.safetensors",
        device="default",
    )

    assert out[0] == {
        "qwen": "qwen_2.5_vl_7b_finetuned_model.safetensors",
        "byt5": "byt5_small.safetensors",
        "device": "default",
    }


def test_ensure_hy_omniweaving_deepstack_support_attaches_mm_in():
    in_layer = torch.nn.Linear(3, 4, dtype=torch.float16)
    diffusion_model = types.SimpleNamespace(mm_in=None, time_in=types.SimpleNamespace(in_layer=in_layer))

    class _Model:
        def __init__(self):
            self.diffusion_model = diffusion_model

        def extra_conds(self, **kwargs):
            return {"base": 1}

    model = _Model()
    patcher = types.SimpleNamespace(model=model)
    patcher.wrappers = []
    patcher.add_wrapper_with_key = lambda wrapper_type, key, wrapper: patcher.wrappers.append((wrapper_type, key, wrapper))
    sd = {
        "mm_in.linear_1.weight": torch.zeros((4, 3)),
        "mm_in.linear_1.bias": torch.zeros((4,)),
        "mm_in.linear_2.weight": torch.zeros((4, 4)),
        "mm_in.linear_2.bias": torch.zeros((4,)),
    }

    attached = runtime_patches.ensure_hy_omniweaving_deepstack_support(patcher, sd)

    assert attached is True
    assert diffusion_model.mm_in is not None
    assert diffusion_model.freeze_main is True
    assert diffusion_model.mm_in.linear_1.weight.dtype == torch.float16
    assert diffusion_model._hy_omniweaving_mm_in_inactive is True
    assert len(patcher.wrappers) == 1
    out = model.extra_conds(all_stack_text_states=torch.tensor([1.0]))
    assert out["base"] == 1
    assert torch.equal(out["all_stack_text_states"].cond, torch.tensor([1.0]))


def test_ensure_hy_omniweaving_deepstack_support_returns_false_without_mm_in_weights():
    diffusion_model = types.SimpleNamespace(mm_in=None)

    class _Model:
        def __init__(self):
            self.diffusion_model = diffusion_model

        def extra_conds(self, **kwargs):
            return {}

    model = _Model()
    patcher = types.SimpleNamespace(model=model)
    patcher.wrappers = []
    patcher.add_wrapper_with_key = lambda wrapper_type, key, wrapper: patcher.wrappers.append((wrapper_type, key, wrapper))

    attached = runtime_patches.ensure_hy_omniweaving_deepstack_support(patcher, {"other.weight": torch.zeros((1,))})

    assert attached is False
    assert diffusion_model.mm_in is None
    assert len(patcher.wrappers) == 1
    assert torch.equal(model.extra_conds(all_stack_text_states=torch.tensor([2.0]))["all_stack_text_states"].cond, torch.tensor([2.0]))


def test_ensure_hy_omniweaving_text_encoder_support_patches_clip_instance():
    clip = _ClipStub(has_byt5=True)

    patched = runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)

    assert patched is True
    assert clip.cond_stage_model._hy_omniweaving_text_encoder_patched is True
    clip.cond_stage_model.set_clip_options({"deepstack": [8, 16], "setclip": True, "crop_start": 2, "task_name": "i2v", "visual_input_count": 1})
    assert clip.cond_stage_model.deepstack_layers == [8, 16]
    assert clip.cond_stage_model.setclip_output is True
    assert clip.cond_stage_model.crop_start_output == 2
    assert clip.cond_stage_model.crop_start_source == "explicit"
    assert clip.cond_stage_model._hy_task_name == "i2v"
    assert clip.cond_stage_model._hy_visual_input_count == 1
    cond, pooled, extra = clip.cond_stage_model.encode_token_weights({"qwen25_7b": [[(151644, 1.0), (151644, 1.0), (1, 1.0), (2, 1.0)]]})
    assert cond == "cond_tensor"
    assert pooled == "pooled_tensor"
    assert "all_stack_text_states" in extra


def test_hy_omniweaving_text_encoder_support_does_not_double_crop_cond_with_explicit_crop_start():
    clip = _ClipStub(has_byt5=True)

    def encode_token_weights(tokens):
        return (
            torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2),
            torch.zeros((1, 2)),
            {},
        )

    clip.cond_stage_model.encode_token_weights = encode_token_weights
    runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)
    clip.cond_stage_model.set_clip_options({"deepstack": [8, 16], "setclip": False, "crop_start": 2})

    cond, _, extra = clip.cond_stage_model.encode_token_weights(
        {"qwen25_7b": [[(151644, 1.0), (151644, 1.0), (11, 1.0), (12, 1.0), (13, 1.0), (14, 1.0)]]}
    )

    assert tuple(cond.shape) == (1, 4, 2)
    assert "attention_mask" not in extra
    assert tuple(extra["all_stack_text_states"].shape) == (3, 1, 4, 2)


def test_hy_omniweaving_text_encoder_support_preserves_nonempty_t2v_cond():
    clip = _ClipStub(has_byt5=True)

    def encode_token_weights(tokens):
        return (
            torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2),
            torch.zeros((1, 2)),
            {},
        )

    def qwen_encode_token_weights(token_weight_pairs):
        qwen_out = torch.ones((1, 3, 116, 2))
        qwen_extra = {"attention_mask": torch.ones((1, 116))}
        return qwen_out, None, qwen_extra

    clip.cond_stage_model.encode_token_weights = encode_token_weights
    clip.cond_stage_model.qwen25_7b.encode_token_weights = qwen_encode_token_weights
    runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)
    clip.cond_stage_model.set_clip_options({"deepstack": [8, 16], "setclip": True, "crop_start": 108, "task_name": "t2v"})

    cond, _, extra = clip.cond_stage_model.encode_token_weights(
        {"qwen25_7b": [[(151644, 1.0)] * 108 + [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0)]]}
    )

    assert tuple(cond.shape) == (1, 8, 2)
    assert tuple(extra["all_stack_text_states"].shape) == (3, 1, 8, 2)


def test_ensure_hy_omniweaving_text_encoder_support_is_idempotent():
    clip = _ClipStub(has_byt5=True)

    assert runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip) is True
    assert runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip) is False


def test_hy_omniweaving_diffusion_wrapper_injects_dit_patch():
    class _Executor:
        def __init__(self):
            self.class_obj = types.SimpleNamespace(
                mm_in=lambda x: torch.ones((2, 1, 3)),
                freeze_main=True,
                double_blocks=[object(), object()],
            )
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "ok"

    executor = _Executor()
    context = torch.zeros((1, 5, 3))
    transformer_options = {}

    out = runtime_patches._hy_omniweaving_diffusion_model_wrapper(
        executor,
        torch.zeros((1, 1)),
        torch.tensor([1.0]),
        context,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        transformer_options,
        all_stack_text_states=torch.zeros((2, 1, 3)),
    )

    assert out == "ok"
    args, kwargs = executor.calls[0]
    patched_options = args[-1]
    assert "patches_replace" in patched_options
    assert "dit" in patched_options["patches_replace"]
    assert ("double_block", 0) in patched_options["patches_replace"]["dit"]


def test_hy_omniweaving_diffusion_wrapper_skips_inactive_connector():
    class _Executor:
        def __init__(self):
            mm_in = lambda x: (_ for _ in ()).throw(AssertionError("inactive mm_in should not run"))
            self.class_obj = types.SimpleNamespace(
                mm_in=mm_in,
                _hy_omniweaving_mm_in_inactive=True,
                freeze_main=True,
                double_blocks=[object(), object()],
            )
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "ok"

    executor = _Executor()
    context = torch.zeros((1, 5, 3))
    transformer_options = {}

    out = runtime_patches._hy_omniweaving_diffusion_model_wrapper(
        executor,
        torch.zeros((1, 1)),
        torch.tensor([1.0]),
        context,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        transformer_options,
        all_stack_text_states=torch.zeros((2, 1, 3)),
    )

    assert out == "ok"
    args, kwargs = executor.calls[0]
    assert args[-1] == {}
    assert kwargs == {}


def test_cond_deepstack_preserves_layer_dim_when_processing_and_concat():
    cond = runtime_patches._CONDDeepstackTextStates(torch.arange(3 * 1 * 5 * 2, dtype=torch.float32).reshape(3, 1, 5, 2))

    processed = cond.process_cond(batch_size=2)

    assert tuple(processed.cond.shape) == (3, 2, 5, 2)
    assert torch.equal(processed.cond[:, 0], cond.cond[:, 0])
    assert torch.equal(processed.cond[:, 1], cond.cond[:, 0])

    combined = processed.concat([processed])
    assert tuple(combined.shape) == (3, 4, 5, 2)
