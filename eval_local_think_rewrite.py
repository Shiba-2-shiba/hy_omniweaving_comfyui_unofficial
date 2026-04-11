import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
WORK_ROOT = REPO_ROOT.parent
DEFAULT_COMFY_ROOT = WORK_ROOT / "参考リポジトリ" / "ComfyUI"
DEFAULT_QWEN_PATH = WORK_ROOT / "qwen_2.5_vl_7b_finetuned_model.safetensors"
DEFAULT_BYT5_NAME = "byt5_small_glyphxl_fp16.safetensors"


def _configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a local OmniWeaving think-rewrite probe using the reference ComfyUI loader/runtime path."
    )
    parser.add_argument("--comfy-root", default=str(DEFAULT_COMFY_ROOT), help="Path to the reference ComfyUI checkout.")
    parser.add_argument("--qwen-path", default=str(DEFAULT_QWEN_PATH), help="Qwen text-encoder safetensors path or basename.")
    parser.add_argument("--byt5-path", default=DEFAULT_BYT5_NAME, help="ByT5 safetensors path or basename.")
    parser.add_argument("--text-encoder-dir", action="append", default=[], help="Extra directory to register for text_encoders.")
    parser.add_argument("--qwen-tokenizer-root", help="Optional tokenizer asset directory to force for Comfy's qwen25 tokenizer.")
    parser.add_argument("--prompt", required=True, help="Prompt to evaluate.")
    parser.add_argument("--task", choices=["t2v", "i2v", "interpolation"], default="i2v")
    parser.add_argument("--image", help="Optional image path for i2v/interpolation rewrite probing.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="default", choices=["default", "cpu"], help="Text encoder load device.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed passed into clip.generate.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--trace-top-k", type=int, default=0, help="When > 0, record per-step token-choice traces with this many top logits.")
    parser.add_argument("--dry-run", action="store_true", help="Validate imports and model path resolution without loading models.")
    parser.add_argument("--debug", action="store_true", help="Enable HY_OMNIWEAVING_DEBUG=1 during the run.")
    return parser.parse_args()


def _normalize_model_arg(value: str):
    path = Path(value)
    if path.exists():
        return path.resolve(), path.name
    return None, value


def _token_summary(tokens):
    if not isinstance(tokens, dict):
        return {"type": type(tokens).__name__}
    summary = {"keys": sorted(tokens.keys())}
    input_ids = tokens.get("input_ids")
    attention_mask = tokens.get("attention_mask")
    if torch.is_tensor(input_ids):
        summary["input_ids_shape"] = tuple(input_ids.shape)
    if torch.is_tensor(attention_mask):
        summary["attention_mask_shape"] = tuple(attention_mask.shape)
        summary["prompt_tokens"] = [int(v) for v in attention_mask.sum(dim=1).tolist()]
    qwen_tokens = tokens.get("qwen25_7b")
    if isinstance(qwen_tokens, list) and len(qwen_tokens) > 0 and isinstance(qwen_tokens[0], list):
        summary["qwen_pair_count"] = len(qwen_tokens[0])
    return summary


def _load_visual_image(path: str):
    image = Image.open(path).convert("RGB")
    data = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
    return data.unsqueeze(0)


def _import_runtime(comfy_root: Path):
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(REPO_ROOT))

    import comfy.options

    comfy.options.enable_args_parsing(False)

    import folder_paths
    import nodes

    return folder_paths, nodes


def _apply_qwen_tokenizer_override(tokenizer_root: str):
    if not tokenizer_root:
        return None

    from transformers import Qwen2Tokenizer
    import comfy.text_encoders.qwen_image as qwen_image
    import comfy.sd1_clip as sd1_clip

    override_root = str(Path(tokenizer_root).resolve())

    class OverriddenQwen25Tokenizer(sd1_clip.SDTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(
                override_root,
                pad_with_end=False,
                embedding_size=3584,
                embedding_key="qwen25_7b",
                tokenizer_class=Qwen2Tokenizer,
                has_start_token=False,
                has_end_token=False,
                pad_to_max_length=False,
                max_length=99999999,
                min_length=1,
                pad_token=151643,
                tokenizer_data=tokenizer_data,
            )

    qwen_image.Qwen25_7BVLITokenizer = OverriddenQwen25Tokenizer
    return override_root


def _register_text_encoder_paths(folder_paths, qwen_path: str, byt5_path: str, extra_dirs: list[str]):
    registered = []
    for value in [qwen_path, byt5_path, *extra_dirs]:
        resolved, _name = _normalize_model_arg(value)
        if resolved is None:
            continue
        folder = str(resolved.parent)
        if folder not in registered:
            folder_paths.add_model_folder_path("text_encoders", folder, is_default=True)
            registered.append(folder)
    return registered


def _resolve_registered_name(folder_paths, value: str):
    resolved, name = _normalize_model_arg(value)
    if resolved is not None:
        folder_paths.get_full_path_or_raise("text_encoders", name)
        return name, str(resolved)
    full_path = folder_paths.get_full_path_or_raise("text_encoders", value)
    return value, full_path


def _build_rewrite_request_prompt(task: str, prompt: str):
    if task == "i2v":
        expand_prefix = "Here is a concise description of the target video starting with the given image: "
        expand_postfix = " Please generate a more detailed description based on the provided image and the short description."
    elif task == "interpolation":
        expand_prefix = "Here is a concise description of how the video transitions from the first image to the second image: "
        expand_postfix = " Please generate a more detailed description of the transition, based on the provided images and the short description."
    else:
        expand_prefix = "Here is a concise description of the target video: "
        expand_postfix = " Please generate a more detailed description based on the short description."
    return f"{expand_prefix}{prompt}{expand_postfix}"


def _install_generation_trace(trace_top_k: int):
    if trace_top_k <= 0:
        return []

    import comfy.text_encoders.llama as llama

    trace_records = []
    original_sample_token = llama.BaseGenerate.sample_token
    original_generate = llama.BaseGenerate.generate

    def traced_sample_token(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history, generator, do_sample=True, presence_penalty=0.0):
        snapshot = logits.detach().float().cpu()
        top_count = min(trace_top_k, snapshot.shape[-1])
        top_vals, top_ids = torch.topk(snapshot, top_count, dim=-1)
        next_token = original_sample_token(
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
        trace_records.append(
            {
                "history_len": len(token_history),
                "chosen_token_id": int(next_token[0].item()),
                "pre_filter_top_ids": [int(v) for v in top_ids[0].tolist()],
                "pre_filter_top_logits": [float(v) for v in top_vals[0].tolist()],
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "do_sample": do_sample,
            }
        )
        return next_token

    def traced_generate(self, embeds=None, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0, seed=42, stop_tokens=None, initial_tokens=[], execution_dtype=None, min_tokens=0, presence_penalty=0.0):
        trace_records.append(
            {
                "event": "generate_start",
                "max_length": max_length,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
                "stop_tokens": list(stop_tokens) if stop_tokens is not None else None,
                "initial_tokens": list(initial_tokens),
                "execution_dtype": str(execution_dtype) if execution_dtype is not None else None,
                "min_tokens": min_tokens,
                "presence_penalty": presence_penalty,
                "embed_shape": tuple(embeds.shape) if torch.is_tensor(embeds) else None,
            }
        )
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

    llama.BaseGenerate.sample_token = traced_sample_token
    llama.BaseGenerate.generate = traced_generate
    return trace_records


def main():
    _configure_stdout()
    args = _parse_args()
    comfy_root = Path(args.comfy_root).resolve()
    if args.debug:
        os.environ["HY_OMNIWEAVING_DEBUG"] = "1"

    folder_paths, nodes = _import_runtime(comfy_root)
    tokenizer_override_root = _apply_qwen_tokenizer_override(args.qwen_tokenizer_root)
    generation_trace = _install_generation_trace(args.trace_top_k)
    registered = _register_text_encoder_paths(folder_paths, args.qwen_path, args.byt5_path, args.text_encoder_dir)
    qwen_name, qwen_full_path = _resolve_registered_name(folder_paths, args.qwen_path)
    byt5_name, byt5_full_path = _resolve_registered_name(folder_paths, args.byt5_path)

    result = {
        "comfy_root": str(comfy_root),
        "repo_root": str(REPO_ROOT),
        "registered_text_encoder_dirs": registered,
        "qwen_name": qwen_name,
        "qwen_full_path": qwen_full_path,
        "byt5_name": byt5_name,
        "byt5_full_path": byt5_full_path,
        "qwen_tokenizer_root": tokenizer_override_root,
        "task": args.task,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "image": args.image,
    }

    if args.dry_run:
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        print(output)
        return

    clip = nodes._load_hy_omniweaving_dual_text_encoder(qwen_name, byt5_name, device=args.device)
    visual_images = []
    image_embeds = []
    if args.image:
        visual_images = [nodes.TextEncodeHunyuanVideo15Omni._extract_visual_images(_load_visual_image(args.image), 1)[0]]

    rewrite_request_prompt = _build_rewrite_request_prompt(args.task, args.prompt)
    think_visual_images = nodes.TextEncodeHunyuanVideo15Omni._prepare_think_visual_images(visual_images)
    think_visual_count = len(think_visual_images) if len(think_visual_images) > 0 else len(image_embeds)
    think_template = nodes.TextEncodeHunyuanVideo15Omni._build_think_template(args.task, think_visual_count)
    tokens = nodes.TextEncodeHunyuanVideo15Omni._tokenize_with_template(
        clip,
        rewrite_request_prompt,
        think_template,
        image_embeds,
        visual_images=think_visual_images if len(think_visual_images) > 0 else None,
    )
    generated = nodes.TextEncodeHunyuanVideo15Omni._generate_with_rewrite_suppression(
        clip,
        tokens,
        min(args.max_new_tokens, nodes.TextEncodeHunyuanVideo15Omni.THINK_MAX_EFFECTIVE_NEW_TOKENS),
    )
    decoded = nodes.TextEncodeHunyuanVideo15Omni._decode_generated_text(clip, generated, tokens)
    rewritten = nodes.TextEncodeHunyuanVideo15Omni._rewrite_prompt_with_think(
        clip,
        args.prompt,
        args.task,
        image_embeds,
        visual_images,
        args.max_new_tokens,
    )

    result.update(
        {
            "rewrite_request_prompt": rewrite_request_prompt,
            "think_template_preview": think_template[:240],
            "token_summary": _token_summary(tokens),
            "generated_type": type(generated).__name__,
            "generated_shape": tuple(generated.shape) if torch.is_tensor(generated) else None,
            "decoded_generated_text": decoded,
            "helper_rewritten_prompt": rewritten,
            "rewrite_changed_prompt": rewritten != args.prompt,
            "visual_input_count": len(visual_images),
        }
    )
    if args.trace_top_k > 0:
        result["generation_trace"] = generation_trace

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
