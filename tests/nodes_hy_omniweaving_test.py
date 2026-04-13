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


def _assert_clip_options_include(clip, expected):
    for kind, options in clip.clip_options:
        if kind != "set" or not isinstance(options, dict):
            continue
        if all(options.get(key) == value for key, value in expected.items()):
            return
    raise AssertionError(f"Expected clip options subset not found: {expected!r} in {clip.clip_options!r}")


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


def test_hy_omniweaving_local_prepared_input_spec_uses_t2v_text_defaults():
    spec = nodes.TextEncodeHunyuanVideo15Omni._prepare_input_local_spec(
        task="t2v",
        prompt="A lighthouse in a storm",
        use_visual_inputs=True,
        max_visual_inputs=8,
    )

    assert isinstance(spec, nodes.LocalPreparedInputSpec)
    assert spec.task == "t2v"
    assert spec.prompt_mode == 1
    assert spec.crop_start == 108
    assert spec.visual_input_count == 0
    assert spec.ordered_roles == []
    assert spec.token_budget_extra == 0
    assert spec.used_fallback_text_only is False
    assert "Describe the video by detailing the following aspects" in spec.template
    assert "<|vision_start|>" not in spec.template


def test_hy_omniweaving_local_prepared_input_spec_prefers_semantic_images_for_i2v():
    reference_images = torch.zeros((1, 640, 640, 3))
    semantic_images = torch.ones((1, 640, 320, 3))

    spec = nodes.TextEncodeHunyuanVideo15Omni._prepare_input_local_spec(
        task="i2v",
        prompt="A dancer starts moving",
        reference_images=reference_images,
        semantic_images=semantic_images,
        use_visual_inputs=True,
        max_visual_inputs=8,
    )

    assert spec.prompt_mode == 2
    assert spec.crop_start == 92
    assert spec.visual_input_count == 1
    assert spec.ordered_roles == ["image"]
    assert spec.meta["visual_source"] == "semantic_images"
    assert tuple(spec.ordered_visuals[0].shape) == (1, 560, 280, 3)
    assert torch.all(spec.ordered_visuals[0] == 1)
    assert spec.token_budget_extra == 400
    assert spec.used_fallback_text_only is False
    assert spec.template.count("<|vision_start|><|image_pad|><|vision_end|>") == 1


def test_hy_omniweaving_local_prepared_input_spec_falls_back_to_text_only_for_i2v_without_visuals():
    spec = nodes.TextEncodeHunyuanVideo15Omni._prepare_input_local_spec(
        task="i2v",
        prompt="A dancer starts moving",
        reference_images=None,
        semantic_images=None,
        use_visual_inputs=True,
        max_visual_inputs=8,
    )

    assert spec.task == "i2v"
    assert spec.prompt_mode == 1
    assert spec.crop_start == 108
    assert spec.visual_input_count == 0
    assert spec.ordered_roles == []
    assert spec.meta["effective_task"] == "t2v"
    assert spec.meta["visual_source"] == "none"
    assert spec.used_fallback_text_only is True
    assert "<|vision_start|>" not in spec.template


def test_hy_omniweaving_local_prepared_input_spec_orders_tiv2v_reference_before_video_frames():
    reference_images = torch.zeros((1, 512, 512, 3))
    video_frames = torch.zeros((2, 320, 480, 3))

    spec = nodes.TextEncodeHunyuanVideo15Omni._prepare_input_local_spec(
        task="tiv2v",
        prompt="Apply the reference style to the motion",
        reference_images=reference_images,
        video_frames=video_frames,
        use_visual_inputs=True,
        max_visual_inputs=8,
    )

    assert spec.prompt_mode == 6
    assert spec.crop_start == 104
    assert spec.visual_input_count == 3
    assert spec.ordered_roles == ["image", "video_frame", "video_frame"]
    assert spec.meta["effective_task"] == "tiv2v"
    assert spec.meta["image_visual_count"] == 1
    assert spec.meta["video_frame_count"] == 2
    assert spec.token_budget_extra == 832
    assert "This is the reference image:" in spec.template
    assert "This is the input video:" in spec.template
    assert spec.template.count("<|vision_start|><|image_pad|><|vision_end|>") == 3


def test_hy_omniweaving_local_prepared_input_spec_falls_back_to_editing_for_tiv2v_with_only_video_frames():
    video_frames = torch.zeros((3, 320, 480, 3))

    spec = nodes.TextEncodeHunyuanVideo15Omni._prepare_input_local_spec(
        task="tiv2v",
        prompt="Apply the reference style to the motion",
        reference_images=None,
        video_frames=video_frames,
        use_visual_inputs=True,
        max_visual_inputs=8,
    )

    assert spec.prompt_mode == 5
    assert spec.crop_start == 90
    assert spec.visual_input_count == 3
    assert spec.ordered_roles == ["video_frame", "video_frame", "video_frame"]
    assert spec.meta["effective_task"] == "editing"
    assert spec.meta["visual_source"] == "video_frames"
    assert spec.used_fallback_text_only is False
    assert "This is the reference image:" not in spec.template
    assert "This is the input video:" not in spec.template


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


def test_encode_hy_omniweaving_redux_clip_vision_output_combines_encoder_and_embedder(monkeypatch):
    class _FakeEncoder:
        def __call__(self, pixel_values=None, output_hidden_states=False):
            batch = pixel_values.shape[0]
            last_hidden = torch.full((batch, 4, 1152), 3.0, dtype=pixel_values.dtype, device=pixel_values.device)
            penultimate = torch.full((batch, 4, 1152), 2.0, dtype=pixel_values.dtype, device=pixel_values.device)
            first = torch.full((batch, 4, 1152), 1.0, dtype=pixel_values.dtype, device=pixel_values.device)
            return types.SimpleNamespace(
                last_hidden_state=last_hidden,
                hidden_states=(first, penultimate, last_hidden),
            )

    class _FakeEmbedder(torch.nn.Module):
        def forward(self, x):
            batch, tokens, _ = x.shape
            out = torch.zeros((batch, tokens, 4096), dtype=x.dtype, device=x.device)
            out[:, :, 0] = 7.0
            return out

    monkeypatch.setattr(
        nodes,
        "_load_hy_omniweaving_redux_vision_models",
        lambda image_encoder_dir, image_embedder_dir, device="default": {
            "encoder": _FakeEncoder(),
            "embedder": _FakeEmbedder(),
            "image_size": 512,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        },
    )

    images = torch.rand((2, 240, 160, 3), dtype=torch.float32)
    output = nodes._encode_hy_omniweaving_redux_clip_vision_output(
        images=images,
        image_encoder_dir="image_encorder",
        image_embedder_dir="image_embedder",
        crop="center",
        device="default",
    )

    assert tuple(output.last_hidden_state.shape) == (2, 4, 1152)
    assert tuple(output.penultimate_hidden_states.shape) == (2, 4, 1152)
    assert tuple(output.all_hidden_states.shape) == (2, 3, 4, 1152)
    assert tuple(output.image_embeds.shape) == (2, 4, 1152)
    assert tuple(output.mm_projected.shape) == (2, 4, 4096)
    assert output.image_sizes == [(3, 512, 512), (3, 512, 512)]
    assert torch.all(output.mm_projected[:, :, 0] == 7.0)


def test_resolve_redux_model_file_accepts_clip_vision_relative_model_file(monkeypatch, tmp_path):
    relative_model = "redux_encoder_test/model.safetensors"
    model_dir = tmp_path / "clip_vision" / "redux_encoder_test"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"stub")

    monkeypatch.setattr(
        nodes.folder_paths,
        "get_full_path",
        lambda folder_name, filename: str(model_dir / "model.safetensors")
        if folder_name == "clip_vision" and filename == relative_model
        else None,
        raising=False,
    )

    resolved = nodes._resolve_redux_model_file(
        relative_model,
        default_filenames=("model.safetensors",),
    )

    assert resolved == str((model_dir / "model.safetensors").resolve())


def test_select_siglip_vision_config_uses_bundled_config_when_shapes_match():
    sd = {
        "vision_model.embeddings.patch_embedding.weight": torch.zeros((1152, 3, 16, 16)),
        "vision_model.embeddings.position_embedding.weight": torch.zeros((1024, 1152)),
        "vision_model.encoder.layers.0.mlp.fc1.weight": torch.zeros((4304, 1152)),
        "vision_model.encoder.layers.0.layer_norm1.weight": torch.zeros((1152,)),
        "vision_model.encoder.layers.26.layer_norm1.weight": torch.zeros((1152,)),
    }

    config = nodes._select_siglip_vision_config(sd)

    assert config["image_size"] == 512
    assert config["patch_size"] == 16
    assert config["num_hidden_layers"] == 27
    assert config["num_attention_heads"] == 16


def test_select_redux_embedder_config_falls_back_to_state_dict_when_bundled_mismatches(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_load_json_file",
        lambda path: {"redux_dim": 999, "txt_in_features": 999},
    )
    sd = {
        "redux_up.weight": torch.zeros((12288, 1152)),
        "redux_down.weight": torch.zeros((4096, 12288)),
    }

    config = nodes._select_redux_embedder_config(sd)

    assert config["redux_dim"] == 1152
    assert config["txt_in_features"] == 4096


def test_hy_omniweaving_redux_vision_encode_node_returns_clip_vision_output(monkeypatch):
    captured = {}
    fake_output = types.SimpleNamespace(mm_projected=torch.zeros((1, 8, 4096)))

    def fake_encode(images, image_encoder_dir, image_embedder_dir, crop="center", device="default"):
        captured["args"] = {
            "shape": tuple(images.shape),
            "image_encoder_dir": image_encoder_dir,
            "image_embedder_dir": image_embedder_dir,
            "crop": crop,
            "device": device,
        }
        return fake_output

    monkeypatch.setattr(nodes, "_encode_hy_omniweaving_redux_clip_vision_output", fake_encode)

    images = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    out = nodes.HYOmniWeavingReduxVisionEncode.execute(
        images=images,
        image_encoder_model="image_encorder/model.safetensors",
        image_embedder_model="image_embedder/diffusion_pytorch_model.safetensors",
        crop="none",
        device="cpu",
    )

    assert out[0] is fake_output
    assert captured["args"] == {
        "shape": (1, 512, 512, 3),
        "image_encoder_dir": "image_encorder/model.safetensors",
        "image_embedder_dir": "image_embedder/diffusion_pytorch_model.safetensors",
        "crop": "none",
        "device": "cpu",
    }


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


def test_hy_omniweaving_text_encode_falls_back_to_reference_images_for_i2v():
    clip = _ClipStub(has_byt5=True)
    reference_images = torch.zeros((1, 640, 640, 3))
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
        semantic_images=None,
        clip_vision_output=clip_vision_output,
    )

    assert "images" in clip.tokenize_calls[0][1]
    assert len(clip.tokenize_calls[0][1]["images"]) == 1
    assert tuple(clip.tokenize_calls[0][1]["images"][0].shape) == (1, 560, 560, 3)


def test_hy_omniweaving_encode_prompt_components_uses_prepared_spec_for_t2v():
    clip = _ClipStub(has_byt5=True)

    encoded = nodes.TextEncodeHunyuanVideo15Omni._encode_prompt_components(
        clip=clip,
        prompt="A lighthouse in a storm",
        task="t2v",
        deepstack_layers="8,16,24",
        setclip=True,
        image_embeds=[],
        visual_images=[],
        visual_source="none",
    )

    assert isinstance(encoded["prepared_spec"], nodes.LocalPreparedInputSpec)
    assert encoded["prepared_spec"].prompt_mode == 1
    assert encoded["crop_start"] == 108
    assert encoded["visual_input_count"] == 0
    assert "Describe the video by detailing the following aspects" in clip.tokenize_calls[0][1]["llama_template"]


def test_hy_omniweaving_encode_prompt_components_uses_prepared_spec_for_i2v_visual_path():
    clip = _ClipStub(has_byt5=True)
    visual_images = [torch.zeros((1, 640, 640, 3))]

    encoded = nodes.TextEncodeHunyuanVideo15Omni._encode_prompt_components(
        clip=clip,
        prompt="A dancer starts moving",
        task="i2v",
        deepstack_layers="8,16,24",
        setclip=True,
        image_embeds=[],
        visual_images=visual_images,
        visual_source="semantic_images",
    )

    assert isinstance(encoded["prepared_spec"], nodes.LocalPreparedInputSpec)
    assert encoded["prepared_spec"].prompt_mode == 2
    assert encoded["crop_start"] == 92
    assert encoded["visual_input_count"] == 1
    assert tuple(clip.tokenize_calls[0][1]["images"][0].shape) == (1, 640, 640, 3)


def test_hy_omniweaving_encode_prompt_components_keeps_legacy_path_for_image_embed_fallback():
    clip = _ClipStub(has_byt5=True)
    image_embeds = [torch.zeros((16, 4096))]

    encoded = nodes.TextEncodeHunyuanVideo15Omni._encode_prompt_components(
        clip=clip,
        prompt="A dancer starts moving",
        task="i2v",
        deepstack_layers="8,16,24",
        setclip=True,
        image_embeds=image_embeds,
        visual_images=[],
        visual_source="clip_vision_output.mm_projected",
    )

    assert encoded["prepared_spec"] is None
    assert encoded["crop_start"] == 92
    assert encoded["visual_input_count"] == 1


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
    assert "Describe the video by detailing the following aspects" in clip.tokenize_calls[0][1]["llama_template"]
    assert clip.last_generate["do_sample"] is False
    assert clip.last_generate["max_length"] == 111
    assert "Here is a more detailed description. expanded prompt" in clip.tokenize_calls[1][0]
    assert "Here is a more detailed description. expanded prompt" in clip.last_encoded["tokens"]
    assert "Describe the video by detailing the following aspects" in clip.tokenize_calls[1][1]["llama_template"]
    _assert_clip_options_include(
        clip,
        {
            "execution_device": "cpu",
            "deepstack": [8, 16, 24],
            "setclip": True,
            "crop_start": 108,
            "task_name": "i2v",
            "visual_input_count": 0,
        },
    )


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
    assert clip.last_generate["max_length"] == 1000


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


def test_hy_omniweaving_rewrite_suppression_sets_and_clears_transformer_tokens():
    class _GenerateClip(_ClipStub):
        def __init__(self):
            super().__init__(has_byt5=True)
            self.transformer = types.SimpleNamespace()
            self.cond_stage_model.qwen25_7b = types.SimpleNamespace(transformer=self.transformer)

        def generate(self, tokens, do_sample=False, max_length=256):
            self.suppressed_during_generate = tuple(getattr(self.transformer, "_hy_suppressed_token_ids", ()))
            return [151645]

    clip = _GenerateClip()

    generated = nodes.TextEncodeHunyuanVideo15Omni._generate_with_rewrite_suppression(clip, {"tokens": "x"}, 8)

    assert generated == [151645]
    assert 151653 in clip.suppressed_during_generate
    assert not hasattr(clip.transformer, "_hy_suppressed_token_ids")


def test_hy_omniweaving_text_encode_merge_hidden_merges_cond_and_deepstack(monkeypatch):
    clip = _ClipStub(has_byt5=True)
    pooled_base = torch.tensor([[1.0, 2.0]])

    monkeypatch.setattr(nodes, "ensure_runtime_patches", lambda: None)
    monkeypatch.setattr(nodes, "ensure_hy_omniweaving_text_encoder_support", lambda clip: None)

    def encode_token_weights(tokens):
        text = tokens["tokens"]
        is_think = text == "expanded prompt"
        seq = 6 if is_think else 4
        value = 2.0 if is_think else 1.0
        pooled = torch.tensor([[9.0, 9.0]]) if is_think else pooled_base
        return (
            torch.full((1, seq, 2), value),
            pooled,
            {
                "attention_mask": torch.ones((1, seq)),
                "all_stack_text_states": torch.full((3, 1, seq, 2), value),
            },
        )

    clip.cond_stage_model.encode_token_weights = encode_token_weights

    out = nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="A dancer starts moving",
        task="t2v",
        use_visual_inputs=False,
        max_visual_inputs=8,
        think=True,
        think_max_new_tokens=128,
        think_mode="merge_hidden",
        think_keep_tokens=3,
        deepstack_layers="8,16,24",
        setclip=True,
        semantic_images=None,
        clip_vision_output=None,
    )

    cond, extra = out[0][0]
    assert len(clip.tokenize_calls) == 3
    assert "Expand the temporal progression only." in clip.tokenize_calls[1][0]
    assert "avoiding unnecessary static scene or appearance description" in clip.tokenize_calls[1][0]
    assert clip.last_generate["do_sample"] is False
    assert clip.last_generate["max_length"] == 128
    assert clip.tokenize_calls[2][0] == "expanded prompt"
    assert tuple(cond.shape) == (1, 7, 2)
    assert tuple(extra["all_stack_text_states"].shape) == (3, 1, 7, 2)
    assert tuple(extra["attention_mask"].shape) == (1, 7)
    assert torch.equal(extra["pooled_output"], pooled_base)


def test_hy_omniweaving_i2v_think_prompt_preserves_first_frame_constraints():
    prompt = nodes.TextEncodeHunyuanVideo15Omni._build_think_conditioning_prompt("i2v", "A girl turns around")

    assert "preserve the same subject identity, background, layout, lighting, and overall framing" in prompt
    assert "first-frame anchoring" in prompt
    assert "camera motion" not in prompt
    assert "new camera setup" in prompt


def test_hy_omniweaving_merge_hidden_rewrite_request_focuses_on_temporal_changes():
    request = nodes.TextEncodeHunyuanVideo15Omni._build_think_rewrite_request(
        "i2v",
        "A girl turns around",
        "merge_hidden",
    )

    assert "Expand only the temporal progression for the video." in request
    assert "Preserve the existing subject identity, clothing, background, lighting, framing, and scene layout." in request
    assert "Do not restate static appearance or background details unless they directly change over time." in request
    assert "Please generate a more detailed description" not in request


def test_hy_omniweaving_merge_hidden_uses_full_generated_branch_by_default():
    think_encoding = {
        "cond": torch.ones((1, 90, 2)),
        "extra": {
            "all_stack_text_states": torch.ones((3, 1, 90, 2)),
        },
    }

    keep_tokens = nodes.TextEncodeHunyuanVideo15Omni._resolve_effective_keep_tokens("i2v", 0, think_encoding)

    assert keep_tokens == 90


def test_hy_omniweaving_merge_hidden_keeps_leading_generated_tokens():
    base_encoding = {
        "cond": torch.full((1, 2, 1), 1.0),
        "pooled_output": torch.zeros((1, 1)),
        "extra": {
            "all_stack_text_states": torch.full((3, 1, 2, 1), 1.0),
            "attention_mask": torch.ones((1, 2)),
            "pooled_output": torch.zeros((1, 1)),
        },
    }
    think_encoding = {
        "cond": torch.tensor([[[10.0], [20.0], [30.0], [40.0]]]),
        "extra": {
            "all_stack_text_states": torch.tensor(
                [
                    [[[10.0], [20.0], [30.0], [40.0]]],
                    [[[11.0], [21.0], [31.0], [41.0]]],
                    [[[12.0], [22.0], [32.0], [42.0]]],
                ]
            ),
            "attention_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        },
        "tokens": {"qwen25_7b": [[(101, 1.0), (102, 1.0), (103, 1.0), (104, 1.0)]]},
    }

    merged = nodes.TextEncodeHunyuanVideo15Omni._merge_encoded_conditioning(
        base_encoding,
        think_encoding,
        task="i2v",
        think_keep_tokens=2,
    )

    assert torch.equal(merged["cond"], torch.tensor([[[1.0], [1.0], [10.0], [20.0]]]))
    assert torch.equal(
        merged["extra"]["all_stack_text_states"],
        torch.tensor(
            [
                [[[1.0], [1.0], [10.0], [20.0]]],
                [[[1.0], [1.0], [11.0], [21.0]]],
                [[[1.0], [1.0], [12.0], [22.0]]],
            ]
        ),
    )


def test_hy_omniweaving_finalized_attention_mask_expands_for_clip_vision_prefix():
    encoded = {
        "cond": torch.zeros((1, 6, 2)),
        "extra": {
            "attention_mask": torch.ones((1, 6)),
            "pooled_output": torch.zeros((1, 1)),
        },
    }
    clip_vision_output = types.SimpleNamespace(last_hidden_state=torch.zeros((1, 1024, 1152)))

    finalized = nodes.TextEncodeHunyuanVideo15Omni._finalize_encoded_components(
        encoded,
        clip_vision_output=clip_vision_output,
    )

    assert tuple(finalized["extra"]["attention_mask"].shape) == (1, 1030)
    assert torch.equal(finalized["extra"]["attention_mask"][:, :1024], torch.ones((1, 1024)))
    assert torch.equal(finalized["extra"]["attention_mask"][:, 1024:], torch.ones((1, 6)))


def test_hy_omniweaving_finalized_attention_mask_skips_clip_vision_prefix_for_t2v():
    encoded = {
        "task": "t2v",
        "cond": torch.zeros((1, 6, 2)),
        "extra": {
            "attention_mask": torch.ones((1, 6)),
            "pooled_output": torch.zeros((1, 1)),
        },
    }
    clip_vision_output = types.SimpleNamespace(last_hidden_state=torch.zeros((1, 1024, 1152)))

    finalized = nodes.TextEncodeHunyuanVideo15Omni._finalize_encoded_components(
        encoded,
        clip_vision_output=clip_vision_output,
    )

    assert tuple(finalized["extra"]["attention_mask"].shape) == (1, 6)
    assert torch.equal(finalized["extra"]["attention_mask"], torch.ones((1, 6)))


def test_hy_omniweaving_finalized_attention_mask_expands_for_clip_vision_and_byt5_prefixes():
    encoded = {
        "cond": torch.zeros((1, 6, 2)),
        "extra": {
            "attention_mask": torch.ones((1, 6)),
            "conditioning_byt5small": torch.zeros((1, 12, 1472)),
            "pooled_output": torch.zeros((1, 1)),
        },
    }
    clip_vision_output = types.SimpleNamespace(last_hidden_state=torch.zeros((1, 1024, 1152)))

    finalized = nodes.TextEncodeHunyuanVideo15Omni._finalize_encoded_components(
        encoded,
        clip_vision_output=clip_vision_output,
    )

    assert tuple(finalized["extra"]["attention_mask"].shape) == (1, 1042)
    assert torch.equal(finalized["extra"]["attention_mask"][:, :1024], torch.ones((1, 1024)))
    assert torch.equal(finalized["extra"]["attention_mask"][:, 1024:1036], torch.ones((1, 12)))
    assert torch.equal(finalized["extra"]["attention_mask"][:, 1036:], torch.ones((1, 6)))


def test_hy_omniweaving_merge_hidden_ignores_trailing_template_control_tokens():
    base_encoding = {
        "cond": torch.full((1, 4, 1), 1.0),
        "pooled_output": torch.zeros((1, 1)),
        "extra": {
            "attention_mask": torch.ones((1, 4)),
            "all_stack_text_states": torch.full((3, 1, 4, 1), 1.0),
            "pooled_output": torch.zeros((1, 1)),
        },
    }
    think_encoding = {
        "cond": torch.arange(1, 8, dtype=torch.float32).reshape(1, 7, 1),
        "pooled_output": torch.ones((1, 1)),
        "extra": {
            "attention_mask": torch.ones((1, 7)),
            "all_stack_text_states": torch.arange(1, 22, dtype=torch.float32).reshape(3, 1, 7, 1),
        },
        "tokens": {
            "qwen25_7b": [[
                (151644, 1.0),
                (999, 1.0),
                (151645, 1.0),
                (198, 1.0),
                (151644, 1.0),
                (872, 1.0),
                (198, 1.0),
                (11, 1.0),
                (12, 1.0),
                (151645, 1.0),
                (198, 1.0),
                (151644, 1.0),
                (77091, 1.0),
                (198, 1.0),
            ]],
        },
    }

    merged = nodes.TextEncodeHunyuanVideo15Omni._merge_encoded_conditioning(
        base_encoding,
        think_encoding,
        task="i2v",
        think_keep_tokens=10,
    )

    assert tuple(merged["cond"].shape) == (1, 6, 1)
    assert torch.equal(merged["cond"][0, -2:, 0], torch.tensor([1.0, 2.0]))
    assert tuple(merged["extra"]["attention_mask"].shape) == (1, 6)
    assert tuple(merged["extra"]["all_stack_text_states"].shape) == (3, 1, 6, 1)


def test_hy_omniweaving_text_encode_merge_hidden_skips_negative_prompt_like_text(monkeypatch):
    clip = _ClipStub(has_byt5=True)

    monkeypatch.setattr(nodes, "ensure_runtime_patches", lambda: None)
    monkeypatch.setattr(nodes, "ensure_hy_omniweaving_text_encoder_support", lambda clip: None)

    def encode_token_weights(tokens):
        return (
            torch.ones((1, 4, 2)),
            torch.zeros((1, 2)),
            {
                "attention_mask": torch.ones((1, 4)),
                "all_stack_text_states": torch.ones((3, 1, 4, 2)),
            },
        )

    clip.cond_stage_model.encode_token_weights = encode_token_weights

    out = nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="low quality, blurry artifacts, watermark, bad anatomy",
        task="t2v",
        use_visual_inputs=False,
        max_visual_inputs=8,
        think=True,
        think_max_new_tokens=128,
        think_mode="merge_hidden",
        think_keep_tokens=3,
        deepstack_layers="8,16,24",
        setclip=True,
        semantic_images=None,
        clip_vision_output=None,
    )

    cond, extra = out[0][0]
    assert len(clip.tokenize_calls) == 1
    assert not hasattr(clip, "last_generate")
    assert tuple(cond.shape) == (1, 4, 2)
    assert tuple(extra["all_stack_text_states"].shape) == (3, 1, 4, 2)


def test_hy_omniweaving_text_encode_merge_hidden_skips_duplicate_merge_when_ar_returns_nothing(monkeypatch):
    clip = _ClipStub(has_byt5=True)

    monkeypatch.setattr(nodes, "ensure_runtime_patches", lambda: None)
    monkeypatch.setattr(nodes, "ensure_hy_omniweaving_text_encoder_support", lambda clip: None)
    monkeypatch.setattr(
        nodes.TextEncodeHunyuanVideo15Omni,
        "_decode_generated_text",
        staticmethod(lambda clip, generated, tokens: ""),
    )

    def encode_token_weights(tokens):
        return (
            torch.ones((1, 4, 2)),
            torch.zeros((1, 2)),
            {
                "attention_mask": torch.ones((1, 4)),
                "all_stack_text_states": torch.ones((3, 1, 4, 2)),
            },
        )

    clip.cond_stage_model.encode_token_weights = encode_token_weights

    out = nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="A dancer starts moving",
        task="t2v",
        use_visual_inputs=False,
        max_visual_inputs=8,
        think=True,
        think_max_new_tokens=128,
        think_mode="merge_hidden",
        think_keep_tokens=3,
        deepstack_layers="8,16,24",
        setclip=True,
        semantic_images=None,
        clip_vision_output=None,
    )

    cond, extra = out[0][0]
    assert len(clip.tokenize_calls) == 2
    assert clip.last_generate["max_length"] == 128
    assert tuple(cond.shape) == (1, 4, 2)
    assert tuple(extra["all_stack_text_states"].shape) == (3, 1, 4, 2)


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

    _assert_clip_options_include(
        clip,
        {
            "execution_device": "cpu",
            "deepstack": [8, 16, 24],
            "setclip": True,
            "crop_start": 92,
            "task_name": "i2v",
            "visual_input_count": 1,
        },
    )


def test_hy_omniweaving_text_encode_passes_t2v_crop_and_setclip_into_clip_options():
    clip = _ClipStub(has_byt5=True)

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="A lighthouse in a storm",
        task="t2v",
        use_visual_inputs=True,
        max_visual_inputs=8,
        think=False,
        think_max_new_tokens=128,
        deepstack_layers="8,16,24",
        setclip=True,
        reference_images=None,
        semantic_images=None,
        clip_vision_output=None,
    )

    _assert_clip_options_include(
        clip,
        {
            "execution_device": "cpu",
            "deepstack": [8, 16, 24],
            "setclip": True,
            "crop_start": 108,
            "task_name": "t2v",
            "visual_input_count": 0,
        },
    )


def test_hy_omniweaving_text_encode_uses_reference_style_task_template():
    clip = _ClipStub(has_byt5=True)
    reference_images = torch.zeros((1, 640, 640, 3))

    nodes.TextEncodeHunyuanVideo15Omni.execute(
        clip=clip,
        prompt="test",
        task="reference2v",
        use_visual_inputs=True,
        max_visual_inputs=8,
        think=False,
        think_max_new_tokens=1000,
        deepstack_layers="8,16,24",
        setclip=True,
        reference_images=reference_images,
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
    assert "ref_latent" not in pos_values
    assert "ref_latent" not in neg_values
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


def test_patch_qwen25_think_generation_uses_lm_head_weight_for_logits(monkeypatch):
    import comfy.text_encoders.llama as llama

    class _BaseGenerate:
        def generate(self, *args, **kwargs):
            return []

    monkeypatch.setattr(llama, "BaseGenerate", _BaseGenerate)
    monkeypatch.setattr(llama, "Qwen25_7BVLI_Config", type("Qwen25_7BVLI_Config", (), {}))

    runtime_patches._patch_qwen25_think_generation()

    embed_tokens = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        comfy_cast_weights=False,
    )
    lm_head = types.SimpleNamespace(
        weight=torch.nn.Parameter(torch.tensor([[0.0, 2.0], [3.0, 0.0]])),
        comfy_cast_weights=False,
    )
    generator = llama.BaseGenerate()
    generator.model = types.SimpleNamespace(embed_tokens=embed_tokens, lm_head=lm_head)

    logits = generator.logits(torch.tensor([[[1.0, 2.0]]], dtype=torch.float32))

    assert tuple(logits.shape) == (1, 1, 2)
    assert torch.equal(logits[0, 0], torch.tensor([4.0, 3.0]))


def test_patch_qwen25_think_generation_adds_qwen_lm_head(monkeypatch):
    import comfy.text_encoders.llama as llama

    class _BaseGenerate:
        def generate(self, *args, **kwargs):
            return []

    class _FakeQwen25_7BVLI:
        def __init__(self, config_dict, dtype, device, operations):
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(hidden_size=3, vocab_size=5, lm_head=False)
            )

    monkeypatch.setattr(llama, "BaseGenerate", _BaseGenerate)
    monkeypatch.setattr(llama, "Qwen25_7BVLI", _FakeQwen25_7BVLI, raising=False)
    monkeypatch.setattr(llama, "Qwen25_7BVLI_Config", type("Qwen25_7BVLI_Config", (), {}))

    runtime_patches._patch_qwen25_think_generation()

    qwen = llama.Qwen25_7BVLI({}, torch.float32, "cpu", types.SimpleNamespace(Linear=torch.nn.Linear))

    assert hasattr(qwen.model, "lm_head")
    assert tuple(qwen.model.lm_head.weight.shape) == (5, 3)
    assert qwen.model.config.lm_head is True


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
    assert clip.cond_stage_model._hy_prepared_input_meta is None
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


def test_hy_omniweaving_text_encoder_support_applies_setclip_to_cond_and_deepstack():
    clip = _ClipStub(has_byt5=True)

    def encode_token_weights(tokens):
        return (
            torch.arange(1 * 7 * 2, dtype=torch.float32).reshape(1, 7, 2),
            torch.zeros((1, 2)),
            {},
        )

    def qwen_encode_token_weights(token_weight_pairs):
        qwen_out = torch.ones((1, 3, 9, 2))
        qwen_extra = {"attention_mask": torch.ones((1, 9))}
        return qwen_out, None, qwen_extra

    clip.cond_stage_model.encode_token_weights = encode_token_weights
    clip.cond_stage_model.qwen25_7b.encode_token_weights = qwen_encode_token_weights
    runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)
    clip.cond_stage_model.set_clip_options({"deepstack": [8, 16], "setclip": True, "crop_start": 2, "task_name": "i2v"})

    cond, _, extra = clip.cond_stage_model.encode_token_weights(
        {"qwen25_7b": [[(151644, 1.0), (151644, 1.0), (11, 1.0), (12, 1.0), (151653, 1.0), (21, 1.0), (22, 1.0), (23, 1.0), (24, 1.0)]]}
    )

    assert tuple(cond.shape) == (1, 4, 2)
    assert tuple(extra["attention_mask"].shape) == (1, 4)
    assert torch.equal(extra["attention_mask"], torch.ones((1, 4)))
    assert tuple(extra["all_stack_text_states"].shape) == (3, 1, 4, 2)


def test_hy_omniweaving_text_encoder_support_logs_attention_mask_reconstruction(monkeypatch, caplog):
    clip = _ClipStub(has_byt5=True)

    def encode_token_weights(tokens):
        return (
            torch.arange(1 * 7 * 2, dtype=torch.float32).reshape(1, 7, 2),
            torch.zeros((1, 2)),
            {},
        )

    def qwen_encode_token_weights(token_weight_pairs):
        qwen_out = torch.ones((1, 3, 9, 2))
        qwen_extra = {"attention_mask": torch.ones((1, 9))}
        return qwen_out, None, qwen_extra

    clip.cond_stage_model.encode_token_weights = encode_token_weights
    clip.cond_stage_model.qwen25_7b.encode_token_weights = qwen_encode_token_weights
    runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)
    clip.cond_stage_model.set_clip_options({"deepstack": [8, 16], "setclip": True, "crop_start": 2, "task_name": "i2v"})

    monkeypatch.setenv("HY_OMNIWEAVING_DEBUG", "1")
    with caplog.at_level("INFO"):
        clip.cond_stage_model.encode_token_weights(
            {"qwen25_7b": [[(151644, 1.0), (151644, 1.0), (11, 1.0), (12, 1.0), (151653, 1.0), (21, 1.0), (22, 1.0), (23, 1.0), (24, 1.0)]]}
        )

    assert "attention_mask_reason=reconstructed_from_qwen_branch" in caplog.text
    assert "orig_attention_mask_state=missing" in caplog.text
    assert "final_attention_mask_state=tensor(1, 4)" in caplog.text
    assert "cond_stage_model_class=SimpleNamespace" in caplog.text
    assert "orig_encode=nodes_hy_omniweaving_test.test_hy_omniweaving_text_encoder_support_logs_attention_mask_reconstruction.<locals>.encode_token_weights" in caplog.text
    assert "qwen_encode=nodes_hy_omniweaving_test.test_hy_omniweaving_text_encoder_support_logs_attention_mask_reconstruction.<locals>.qwen_encode_token_weights" in caplog.text
    assert "attention_mask_state=tensor(1, 4)" in caplog.text
    assert "attention_mask reconstructed task=i2v cond_stage_model_class=SimpleNamespace source=qwen_branch shape=(1, 4)" in caplog.text


def test_hy_omniweaving_text_encoder_support_stores_prepared_meta_and_clears_on_reset():
    clip = _ClipStub(has_byt5=True)

    runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)
    prepared_meta = {
        "task": "i2v",
        "prompt_mode": 2,
        "crop_start": 92,
        "visual_input_count": 1,
        "used_fallback_text_only": False,
    }
    clip.cond_stage_model.set_clip_options(
        {
            "deepstack": [8, 16],
            "setclip": True,
            "crop_start": 92,
            "task_name": "i2v",
            "visual_input_count": 1,
            "prepared_meta": prepared_meta,
        }
    )

    assert clip.cond_stage_model._hy_prepared_input_meta == prepared_meta

    clip.cond_stage_model.reset_clip_options()

    assert clip.cond_stage_model._hy_prepared_input_meta is None


def test_hy_omniweaving_text_encoder_support_prefers_prepared_meta_crop_source():
    clip = _ClipStub(has_byt5=True)

    def encode_token_weights(tokens):
        return (
            torch.arange(1 * 6 * 2, dtype=torch.float32).reshape(1, 6, 2),
            torch.zeros((1, 2)),
            {},
        )

    clip.cond_stage_model.encode_token_weights = encode_token_weights
    runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)
    clip.cond_stage_model.set_clip_options(
        {
            "deepstack": [],
            "setclip": False,
            "crop_start": 2,
            "task_name": "t2v",
            "prepared_meta": {"crop_start": 4, "used_fallback_text_only": False},
        }
    )

    clip.cond_stage_model.encode_token_weights(
        {"qwen25_7b": [[(151644, 1.0), (151644, 1.0), (11, 1.0), (12, 1.0), (13, 1.0), (14, 1.0)]]}
    )

    assert clip.cond_stage_model.crop_start_source == "prepared_meta"


def test_hy_omniweaving_text_encoder_support_marks_prepared_text_only_setclip_source():
    clip = _ClipStub(has_byt5=True)

    def encode_token_weights(tokens):
        return (
            torch.arange(1 * 6 * 2, dtype=torch.float32).reshape(1, 6, 2),
            torch.zeros((1, 2)),
            {},
        )

    clip.cond_stage_model.encode_token_weights = encode_token_weights
    runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip)
    clip.cond_stage_model.set_clip_options(
        {
            "deepstack": [],
            "setclip": True,
            "crop_start": 108,
            "task_name": "i2v",
            "prepared_meta": {"crop_start": 108, "used_fallback_text_only": True},
        }
    )

    clip.cond_stage_model.encode_token_weights(
        {"qwen25_7b": [[(151644, 1.0), (151644, 1.0), (11, 1.0), (12, 1.0), (13, 1.0), (14, 1.0)]]}
    )

    assert clip.cond_stage_model.setclip_start_source == "prepared_text_only"


def test_ensure_hy_omniweaving_text_encoder_support_is_idempotent():
    clip = _ClipStub(has_byt5=True)

    assert runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip) is True
    assert runtime_patches.ensure_hy_omniweaving_text_encoder_support(clip) is False


def test_ensure_hy_omniweaving_txt_mask_alignment_support_uses_trailing_text_mask():
    recorded = {}

    class _TxtIn:
        def forward(self, x, timesteps, mask, transformer_options=None):
            recorded["x_shape"] = tuple(x.shape)
            recorded["mask_shape"] = tuple(mask.shape) if torch.is_tensor(mask) else None
            recorded["mask"] = mask.clone() if torch.is_tensor(mask) else mask
            return x

    diffusion_model = types.SimpleNamespace(txt_in=_TxtIn())

    patched = runtime_patches._ensure_hy_omniweaving_txt_mask_alignment_support(diffusion_model)
    assert patched is True

    x = torch.zeros((1, 1140, 3584))
    mask = torch.cat([torch.ones((1, 1024)), torch.arange(1140, dtype=torch.float32).reshape(1, 1140)], dim=1)
    out = diffusion_model.txt_in.forward(x, torch.tensor([1.0]), mask, transformer_options={})

    assert out is x
    assert recorded["x_shape"] == (1, 1140, 3584)
    assert recorded["mask_shape"] == (1, 1140)
    assert torch.equal(recorded["mask"], mask[:, -1140:])


def test_ensure_hy_omniweaving_txt_mask_alignment_support_trims_non_prefix_mismatch_for_runtime_safety():
    recorded = {}

    class _TxtIn:
        def forward(self, x, timesteps, mask, transformer_options=None):
            recorded["mask_shape"] = tuple(mask.shape) if torch.is_tensor(mask) else None
            recorded["mask"] = mask.clone() if torch.is_tensor(mask) else mask
            return x

    diffusion_model = types.SimpleNamespace(txt_in=_TxtIn())

    patched = runtime_patches._ensure_hy_omniweaving_txt_mask_alignment_support(diffusion_model)
    assert patched is True

    x = torch.zeros((1, 4, 3584))
    mask = torch.tensor([[0.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    diffusion_model.txt_in.forward(x, torch.tensor([1.0]), mask, transformer_options={})

    assert recorded["mask_shape"] == (1, 4)
    assert torch.equal(recorded["mask"], mask[:, -4:])


def test_ensure_hy_omniweaving_forward_orig_txt_mask_debug_support_preserves_call(monkeypatch, caplog):
    recorded = {}

    class _DiffusionModel:
        def forward_orig(self, *args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return "ok"

    diffusion_model = _DiffusionModel()

    patched = runtime_patches._ensure_hy_omniweaving_forward_orig_txt_mask_debug_support(diffusion_model)
    assert patched is True

    txt_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.int64)
    monkeypatch.setenv("HY_OMNIWEAVING_DEBUG", "1")
    with caplog.at_level("INFO"):
        out = diffusion_model.forward_orig(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            txt_mask,
            torch.tensor([1.0]),
        )

    assert out == "ok"
    assert recorded["args"][4] is txt_mask
    assert "forward_orig txt_mask shape=(1, 4)" in caplog.text
    assert "is_floating=False" in caplog.text
    assert "will_apply_non_floating_conversion=True" in caplog.text


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
