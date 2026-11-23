#!/usr/bin/env python3
"""Export a Laya MultimodalAgent checkpoint and its media encoders to GGUF."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor

sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf

from conversion import ModelBase, ModelType, get_model_architecture, get_model_class
from conversion.base import LazyTorchTensor


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def resolve_backbone(config, override):
    if override is not None:
        return override.resolve(strict=True)
    backbone = Path(config["backbone"])
    if backbone.is_dir():
        return backbone.resolve()
    if backbone.is_absolute():
        raise SystemExit(f"Backbone not found: {backbone}. Supply its directory with --base.")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(config["backbone"], revision=config.get("revision"),
                                  allow_patterns=["*.json", "*.txt", "*.jinja", "*.model", "*.safetensors"]))


def load_adapter(checkpoint):
    config = read_json(checkpoint / "adapter" / "adapter_config.json")
    unsupported = ("use_dora", "use_rslora", "lora_bias", "fan_in_fan_out", "rank_pattern", "alpha_pattern")
    if (config["peft_type"] != "LORA" or config.get("bias", "none") != "none"
            or any(config.get(key) for key in unsupported) or set(config.get("modules_to_save") or ()) - {"lm_head"}):
        raise ValueError("Laya export requires a standard linear LoRA adapter")
    adapter = {}
    for name, value in load_file(checkpoint / "adapter" / "adapter_model.safetensors").items():
        name = name.removeprefix("base_model.model.")
        if name in {"lm_head.weight", "lm_head.modules_to_save.weight"}:
            continue  # The authoritative decision projection is saved separately.
        module, _, part = name.partition(".lora_")
        if part not in {"A.weight", "B.weight"}:
            raise ValueError(f"Unexpected Laya adapter tensor: {name}")
        adapter.setdefault(module + ".weight", {})[part[0]] = value.float()
    return adapter, config["lora_alpha"] / config["r"]


def merge_weight(source, factors, scale):
    weight = source()
    a, b = factors["A"], factors["B"]
    if weight.shape != (b.shape[0], a.shape[1]):
        raise ValueError("Laya LoRA tensor shape differs from the backbone")
    delta = LazyTorchTensor.from_eager((b @ a) * scale)
    return (weight.float() + delta).to(weight.dtype)


def qwen_omni_tensor_key(name):
    name = name.removeprefix("thinker.")
    if name.startswith(("model.visual.", "model.audio_tower.")):
        return name.removeprefix("model.")
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    return name


def set_weights(converter, adapter, scale, projectors, used, qwen_omni):
    for source, name in converter._source_tensor_names.items():
        if name is None:
            continue
        key = qwen_omni_tensor_key(source) if qwen_omni else source
        if key in projectors:
            converter.model_tensors[name] = lambda key=key: projectors[key]
            used.add(key)
        if key in adapter:
            if key in projectors:
                raise ValueError(f"Laya tensor is both adapted and replaced: {key}")
            factors = adapter[key]
            if set(factors) != {"A", "B"}:
                raise ValueError(f"Incomplete Laya LoRA factors: {key}")
            original = converter.model_tensors[name]
            converter.model_tensors[name] = lambda original=original, factors=factors: merge_weight(original, factors, scale)
            used.add(key)


def add_decision_weights(converter, checkpoint, config):
    writer = converter.gguf_writer
    readout = config.get("readout", "letter")
    if readout == "candidate":
        head = load_file(checkpoint / "candidate_head.safetensors")
        weight = torch.cat((head["query.weight"], head["candidate.weight"])).float()
        if head["query.weight"].shape != head["candidate.weight"].shape:
            raise ValueError("Laya candidate projections must have the same shape")
        scale = float(head["log_scale"].float().clamp(max=math.log(100)).exp())
        config["scale"] = scale
    elif readout == "letter":
        weight = load_file(checkpoint / "decision_head.safetensors")["weight"].float()
        if len(config["symbols"]) != weight.shape[0]:
            raise ValueError("Laya decision symbols do not match the projection")
    else:
        raise ValueError(f"Unknown Laya decision readout: {readout}")

    writer.add_tensor(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DECISION_READOUT] + ".weight", weight.contiguous().numpy())
    n_outputs = weight.shape[0]
    if (config.get("grounding") or {}).get("trained"):
        head = load_file(checkpoint / "grounding_head.safetensors")
        for layer, tensor in [(0, gguf.MODEL_TENSOR.DECISION_BOX_NORM), (1, gguf.MODEL_TENSOR.DECISION_BOX_HIDDEN), (3, gguf.MODEL_TENSOR.DECISION_BOX_OUTPUT)]:
            for suffix in ("weight", "bias"):
                writer.add_tensor(f"{gguf.TENSOR_NAMES[tensor]}.{suffix}", head[f"single.{layer}.{suffix}"].float().contiguous().numpy())
        config["box_hidden"] = head["single.1.weight"].shape[0]
        n_outputs += 5
    config["readout"] = readout
    config["projection_size"] = weight.shape[0]
    writer.add_embedding_length_out(n_outputs)


def add_decision_metadata(converter, processor, config):
    sentinel = "LayaDecisionInputSentinel"
    messages = [{"role": "user", "content": [{"type": "text", "text": sentinel}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if text.count(sentinel) != 1:
        raise ValueError("Laya processor must render the user text once")
    prefix, suffix = text.split(sentinel)
    template = "{{ " + json.dumps(prefix) + " }}{% for image in images %}{{ image }}{% endfor %}{{ prompt }}{{ " + json.dumps(suffix) + " }}"
    # The processor can add BOS even when the rendered prompt already has one.
    raw_ids = processor.tokenizer(text, add_special_tokens=False).input_ids
    config["add_special"] = processor(text=text).input_ids[0] != raw_ids
    image = processor.image_processor
    if hasattr(image, "max_soft_tokens"):
        config["image_min_tokens"] = config["image_max_tokens"] = image.max_soft_tokens
    elif hasattr(image, "merge_size") and "shortest_edge" in image.size:
        area = (image.patch_size * image.merge_size) ** 2
        config["image_min_tokens"] = image.size["shortest_edge"] // area
        config["image_max_tokens"] = image.size["longest_edge"] // area
    video = getattr(processor, "video_processor", None)
    if hasattr(video, "max_video_tokens"):
        config["video_max_tokens_per_frame"] = video.max_video_tokens

    types = ("text", "image", "audio", "video")
    profiles = config.get("modality_calibration", {})
    config["modality_calibration"] = {kind: {key: profiles[kind][key] for key in ("temperature", "temperature_by_options", "fitted_types")
                                            if key in profiles[kind]} for kind in types if kind in profiles}
    fields = ("readout", "prompt_version", "symbols", "scale", "projection_size", "box_hidden", "add_special", "max_len", "frames",
              "image_min_tokens", "image_max_tokens", "video_max_tokens_per_frame",
              "use_audio_in_video", "max_audio_seconds", "temperature", "temperature_by_options", "modality_calibration", "ordinal_calibration")
    writer = converter.gguf_writer
    writer.add_decision_type(gguf.DecisionType.LAYA_MULTIMODAL)
    writer.add_string(f"{writer.arch}.decision.laya_config", json.dumps({k: config[k] for k in fields if k in config}, allow_nan=False))
    writer.add_chat_template([{"name": "systemone", "template": template}])


def export(checkpoint, out_dir, outtype, base_override):
    config = read_json(checkpoint / "multimodal_config.json")
    if config["version"] != 1 or config["mode"] not in {"head", "lora", "full"}:
        raise ValueError("Unsupported Laya multimodal checkpoint format")
    full = config["mode"] == "full"
    base = checkpoint / "backbone_config" if full and base_override is None else resolve_backbone(config, base_override)
    processor = AutoProcessor.from_pretrained(checkpoint / "processor", trust_remote_code=False)
    adapter, scale = load_adapter(checkpoint) if config["mode"] == "lora" else ({}, 0.0)
    projectors = load_file(checkpoint / "native_projectors.safetensors") if config.get("native_trainable_modules") and config["mode"] != "full" else {}
    weights = checkpoint if full else base
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}-", dir=out_dir.parent) as staged:
        stage = Path(staged) / "checkpoint"
        stage.mkdir()
        output = stage.parent / "gguf"
        output.mkdir()
        files = {path.name: path for path in base.iterdir() if path.is_file() and not path.name.startswith("model")}
        files.update({path.name: path for path in weights.glob("model*.safetensors*")})
        files.update({path.name: path for path in (checkpoint / "processor").iterdir() if path.is_file()})
        for name, path in files.items():
            (stage / name).symlink_to(path.resolve())

        def make_converter(kind, filename):
            hparams = ModelBase.load_hparams(stage, False)
            if hparams.get("model_type") == "qwen2_5_omni_thinker":
                hparams = {"architectures": ["Qwen2_5OmniModel"], "model_type": "qwen2_5_omni",
                           "thinker_config": hparams, "text_config": hparams["text_config"]}
            architecture = get_model_architecture(hparams, kind)
            cls = get_model_class(architecture, mmproj=kind == ModelType.MMPROJ)
            return cls(stage, outtype, output / filename, hparams=hparams, model_name=out_dir.name)

        text = make_converter(ModelType.TEXT, "model.gguf")
        media = make_converter(ModelType.MMPROJ, "mmproj.gguf")
        qwen_omni = media.global_config.get("model_type") == "qwen2_5_omni"
        if qwen_omni:
            media.gguf_writer.add_audio_chunk_size(2 * media.hparams_audio["n_window"])
            adapter = {qwen_omni_tensor_key(name): value for name, value in adapter.items()}
            projectors = {qwen_omni_tensor_key(name): value for name, value in projectors.items()}
        if not media.model_tensors:
            raise ValueError("No media encoder weights were mapped to mmproj.gguf")
        used = set()
        for converter in (text, media):
            for source, name in converter._source_tensor_names.items():
                source = source.removeprefix("thinker.")
                if name and (source.endswith("lm_head.weight") or source.startswith(("decision_scorer.", "grounding_head."))):
                    converter.model_tensors.pop(name, None)
            set_weights(converter, adapter, scale, projectors, used, qwen_omni)
        if unused := (set(adapter) | set(projectors)) - used:
            raise ValueError(f"Laya weights were not mapped to the GGUF models: {sorted(unused)}")
        add_decision_weights(text, checkpoint, config)
        add_decision_metadata(text, processor, config)
        text.write()
        media.write()
        os.rename(output, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="local base model directory, if the original path has moved")
    parser.add_argument("--outtype", choices=("bf16", "f16", "q8_0"), default="bf16")
    args = parser.parse_args()
    types = {"bf16": gguf.LlamaFileType.MOSTLY_BF16, "f16": gguf.LlamaFileType.MOSTLY_F16, "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}
    logging.basicConfig(level=logging.INFO)
    with torch.inference_mode():
        export(args.checkpoint.resolve(strict=True), args.out_dir, types[args.outtype], args.base)


if __name__ == "__main__":
    main()
