import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
WORK_ROOT = REPO_ROOT.parent
DEFAULT_PROCESSOR_ROOT = WORK_ROOT / "Qwen2.5-VL-7B-Instruct"


def _configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare current HY-OmniWeaving prepared-spec text inputs against source-like processor inputs."
    )
    parser.add_argument("--processor-root", default=str(DEFAULT_PROCESSOR_ROOT), help="Path to Qwen2.5-VL processor assets.")
    parser.add_argument("--prompt", required=True, help="Prompt to compare.")
    parser.add_argument("--task", choices=["t2v", "i2v", "reference2v", "interpolation"], default="i2v")
    parser.add_argument("--image", action="append", default=[], help="Image path(s) used for i2v/reference2v/interpolation.")
    parser.add_argument("--max-visual-inputs", type=int, default=8)
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def _load_visual_image(path: str):
    image = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
    return tensor.unsqueeze(0)


def _import_nodes():
    sys.path.insert(0, str(REPO_ROOT))
    import nodes
    return nodes


def _source_like_messages(task: str, prompt: str, pil_images):
    system_prompts = {
        "t2v": "You are a helpful assistant. Describe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video.",
        "i2v": "You are a helpful assistant. Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter the image to introduce motion and evolution over time. Generate a video using this image as the first frame that meets the user's requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency.",
        "reference2v": "You are a helpful assistant. Given a text instruction and one or more input images, you need to explain how to extract and combine key information from the input images to construct a new image as the video's first frame, and then how the user's text instruction should alter the image to introduce motion and evolution over time. Generate a video that meets the user's requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency.",
        "interpolation": "You are a helpful assistant. Given a text instruction, an image as the first frame of the video, and another image as the last frame of the video, you need to analyze the visual trajectory required to transition from the start to the end. Determine how the elements in the first frame must evolve, move, or transform to align with the last frame based on the text instruction. Generate a video that seamlessly connects these two frames, ensuring the motion and evolution between them fulfill the text description while maintaining temporal consistency",
    }
    if task == "t2v" or len(pil_images) == 0:
        return [[system_prompts["t2v"], {"role": "user", "content": [{"type": "text", "text": prompt or " "}]}]]
    if task == "i2v":
        return [[system_prompts["i2v"], {"role": "user", "content": [{"type": "image", "image": pil_images[0]}, {"type": "text", "text": prompt or " "}]}]]
    if task in ("reference2v", "interpolation"):
        return [[system_prompts[task], {"role": "user", "content": ([{"type": "image", "image": img} for img in pil_images] + [{"type": "text", "text": prompt or " "}])}]]
    raise ValueError(f"Unsupported task: {task}")


def _processor_summary(processor_root: Path, messages):
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(str(processor_root))
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    return {
        "chat_template_preview": texts[0][:320] if len(texts) > 0 else None,
        "input_ids_shape": tuple(input_ids.shape) if torch.is_tensor(input_ids) else None,
        "attention_mask_shape": tuple(attention_mask.shape) if torch.is_tensor(attention_mask) else None,
        "prompt_tokens": [int(v) for v in attention_mask.sum(dim=1).tolist()] if torch.is_tensor(attention_mask) else None,
        "pixel_values_shape": tuple(inputs["pixel_values"].shape) if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]) else None,
        "image_grid_thw_shape": tuple(inputs["image_grid_thw"].shape) if "image_grid_thw" in inputs and torch.is_tensor(inputs["image_grid_thw"]) else None,
    }


def main():
    _configure_stdout()
    args = _parse_args()
    nodes = _import_nodes()

    loaded_images = [_load_visual_image(path) for path in args.image]
    stacked_images = torch.cat(loaded_images, dim=0) if len(loaded_images) > 0 else None
    spec = nodes.TextEncodeHunyuanVideo15Omni._prepare_input_local_spec(
        task=args.task,
        prompt=args.prompt,
        reference_images=stacked_images,
        semantic_images=stacked_images if args.task == "i2v" else None,
        use_visual_inputs=True,
        max_visual_inputs=args.max_visual_inputs,
    )
    pil_images = []
    for tensor in spec.ordered_visuals:
        arr = (tensor[0].clamp(0, 1).mul(255).byte().cpu().numpy())
        pil_images.append(Image.fromarray(arr))

    result = {
        "task": args.task,
        "prompt": args.prompt,
        "local_spec": {
            "prompt_mode": spec.prompt_mode,
            "crop_start": spec.crop_start,
            "visual_input_count": spec.visual_input_count,
            "ordered_roles": spec.ordered_roles,
            "used_fallback_text_only": spec.used_fallback_text_only,
            "token_budget_extra": spec.token_budget_extra,
            "visual_source": spec.meta.get("visual_source"),
            "template_preview": spec.template[:320],
            "visual_shapes": [tuple(v.shape) for v in spec.ordered_visuals],
        },
        "source_like_processor": _processor_summary(
            Path(args.processor_root),
            _source_like_messages(args.task, args.prompt, pil_images),
        ),
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
