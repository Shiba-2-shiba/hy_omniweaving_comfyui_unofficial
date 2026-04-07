import sys
import types

import torch


def _install_global_test_stubs():
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

    comfy = sys.modules.get("comfy")
    if comfy is None:
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        sys.modules["comfy"] = comfy

    if "comfy.clip_vision" not in sys.modules:
        clip_vision = types.ModuleType("comfy.clip_vision")
        clip_vision.Output = type("Output", (), {})
        sys.modules["comfy.clip_vision"] = clip_vision
        comfy.clip_vision = clip_vision

    if "comfy.model_management" not in sys.modules:
        model_management = types.ModuleType("comfy.model_management")
        model_management.is_amd = lambda: False
        model_management.dtype_size = lambda dtype: 1
        model_management.vae_device = lambda: "cpu"
        model_management.vae_offload_device = lambda: "cpu"
        model_management.vae_dtype = lambda device, dtypes: torch.float32
        model_management.intermediate_device = lambda: "cpu"
        model_management.archive_model_dtypes = lambda model: None
        model_management.text_encoder_device = lambda: "cpu"
        model_management.text_encoder_offload_device = lambda: "cpu"
        model_management.text_encoder_dtype = lambda device: torch.float32
        model_management.text_encoder_initial_device = lambda load_device, offload_device, size: load_device
        model_management.supports_cast = lambda device, dtype: True
        model_management.load_models_gpu = lambda *args, **kwargs: None
        sys.modules["comfy.model_management"] = model_management
        comfy.model_management = model_management

    if "comfy.model_patcher" not in sys.modules:
        model_patcher = types.ModuleType("comfy.model_patcher")

        class _Patcher:
            def __init__(self, model=None, load_device=None, offload_device=None):
                self.model = model
                self.load_device = load_device
                self.offload_device = offload_device

            def is_dynamic(self):
                return False

            def clone(self, disable_dynamic=False):
                return self

            def set_model_compute_dtype(self, dtype):
                return None

        model_patcher.CoreModelPatcher = _Patcher
        model_patcher.ModelPatcher = _Patcher
        sys.modules["comfy.model_patcher"] = model_patcher
        comfy.model_patcher = model_patcher

    if "comfy.utils" not in sys.modules:
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
        utils.load_torch_file = lambda path, safe_load=True, return_metadata=False: {} if not return_metadata else ({}, {})
        sys.modules["comfy.utils"] = utils
        comfy.utils = utils

    if "comfy.sd" not in sys.modules:
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

    if "comfy.text_encoders" not in sys.modules:
        text_encoders = types.ModuleType("comfy.text_encoders")
        text_encoders.__path__ = []
        sys.modules["comfy.text_encoders"] = text_encoders
        comfy.text_encoders = text_encoders

    if "comfy.text_encoders.hunyuan_image" not in sys.modules:
        hunyuan_image = types.ModuleType("comfy.text_encoders.hunyuan_image")

        class HunyuanImageTEModel:
            def __init__(self, *args, **kwargs):
                pass

            def encode_token_weights(self, token_weight_pairs):
                return torch.zeros((1, 1, 1)), torch.zeros((1, 1)), {}

            def set_clip_options(self, options):
                return None

            def reset_clip_options(self):
                return None

        hunyuan_image.HunyuanImageTEModel = HunyuanImageTEModel
        sys.modules["comfy.text_encoders.hunyuan_image"] = hunyuan_image
        sys.modules["comfy.text_encoders"].hunyuan_image = hunyuan_image

    if "comfy.text_encoders.llama" not in sys.modules:
        llama = types.ModuleType("comfy.text_encoders.llama")
        llama.Qwen25_7BVLI_Config = type("Qwen25_7BVLI_Config", (), {})

        class BaseGenerate:
            def generate(self, *args, **kwargs):
                return []

        llama.BaseGenerate = BaseGenerate
        sys.modules["comfy.text_encoders.llama"] = llama
        sys.modules["comfy.text_encoders"].llama = llama

    if "comfy.model_detection" not in sys.modules:
        model_detection = types.ModuleType("comfy.model_detection")
        model_detection.detect_unet_config = lambda state_dict, key_prefix, metadata=None: {"image_model": "hunyuan_video"}
        sys.modules["comfy.model_detection"] = model_detection
        comfy.model_detection = model_detection

    if "comfy.conds" not in sys.modules:
        conds = types.ModuleType("comfy.conds")
        conds.CONDRegular = lambda value: value
        sys.modules["comfy.conds"] = conds
        comfy.conds = conds

    if "comfy.model_base" not in sys.modules:
        model_base = types.ModuleType("comfy.model_base")

        class _HV15:
            def extra_conds(self, **kwargs):
                return {}

        class _HV15SR:
            def extra_conds(self, **kwargs):
                return {}

        model_base.HunyuanVideo15 = _HV15
        model_base.HunyuanVideo15_SR_Distilled = _HV15SR
        sys.modules["comfy.model_base"] = model_base
        comfy.model_base = model_base

    if "comfy.ldm" not in sys.modules:
        ldm = types.ModuleType("comfy.ldm")
        ldm.__path__ = []
        sys.modules["comfy.ldm"] = ldm
        comfy.ldm = ldm

    if "comfy.ldm.hunyuan_video" not in sys.modules:
        hv_pkg = types.ModuleType("comfy.ldm.hunyuan_video")
        hv_pkg.__path__ = []
        sys.modules["comfy.ldm.hunyuan_video"] = hv_pkg

    if "comfy.ldm.hunyuan_video.model" not in sys.modules:
        hv_model = types.ModuleType("comfy.ldm.hunyuan_video.model")

        class HunyuanVideo:
            def __init__(self, *args, **kwargs):
                self.params = types.SimpleNamespace(context_in_dim=1)
                self.hidden_size = 1
                self.double_blocks = []
                self.mm_in = None

            def _forward(self, *args, **kwargs):
                return {}

        hv_model.HunyuanVideo = HunyuanVideo
        hv_model.HunyuanVideoParams = type("HunyuanVideoParams", (), {"__dataclass_fields__": {}})
        sys.modules["comfy.ldm.hunyuan_video.model"] = hv_model

    if "comfy.ldm.models" not in sys.modules:
        models_pkg = types.ModuleType("comfy.ldm.models")
        models_pkg.__path__ = []
        sys.modules["comfy.ldm.models"] = models_pkg

    if "comfy.ldm.models.autoencoder" not in sys.modules:
        autoencoder = types.ModuleType("comfy.ldm.models.autoencoder")

        class AutoencodingEngine:
            def __init__(self, *args, **kwargs):
                pass

        class AutoencodingEngineLegacy(AutoencodingEngine):
            def __init__(self, embed_dim, ddconfig, **kwargs):
                super().__init__()

        autoencoder.AutoencodingEngine = AutoencodingEngine
        autoencoder.AutoencodingEngineLegacy = AutoencodingEngineLegacy
        sys.modules["comfy.ldm.models.autoencoder"] = autoencoder

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
            Image=types.SimpleNamespace(Input=_InputFactory.Input),
            Latent=types.SimpleNamespace(Output=_OutputFactory.Output),
        )

        latest.ComfyExtension = type("ComfyExtension", (), {})
        latest.io = io
        sys.modules["comfy_api"] = comfy_api
        sys.modules["comfy_api.latest"] = latest


_install_global_test_stubs()
