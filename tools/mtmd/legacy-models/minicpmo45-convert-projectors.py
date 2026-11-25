"""Convert the split MiniCPM-o 4.5 encoder GGUFs to a libmtmd projector."""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gguf-py"))
import gguf


def get_field(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"Missing GGUF metadata: {key}")
    return field.contents()


def add_audio(writer: gguf.GGUFWriter, reader: gguf.GGUFReader) -> None:
    if get_field(reader, "general.architecture") != "whisper":
        raise ValueError("Expected the MiniCPM-o 4.5 Whisper encoder GGUF")

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    layers = get_field(reader, "encoder_layers")
    if not isinstance(layers, int) or not 0 < layers <= len(tensors):
        raise ValueError(f"Invalid audio encoder layer count: {layers}")

    names = {"encoder.positional_embedding": "a.position_embd.weight"}
    for kind in ("weight", "bias"):
        for i in (1, 2):
            names[f"encoder.conv{i}.{kind}"] = f"a.conv1d.{i}.{kind}"
            names[f"audio_projector.linear{i}.{kind}"] = f"mm.a.mlp.{i}.{kind}"
        names[f"encoder.ln_post.{kind}"] = f"a.post_ln.{kind}"
    for layer in range(layers):
        names[f"encoder.blocks.{layer}.attn.key.weight"] = f"a.blk.{layer}.attn_k.weight"
        for old, new in (
            ("attn.query", "attn_q"), ("attn.value", "attn_v"), ("attn.out", "attn_out"),
            ("attn_ln", "ln1"), ("mlp_ln", "ln2"), ("mlp.0", "ffn_up"), ("mlp.2", "ffn_down"),
        ):
            for kind in ("weight", "bias"):
                names[f"encoder.blocks.{layer}.{old}.{kind}"] = f"a.blk.{layer}.{new}.{kind}"

    if names.keys() != tensors.keys():
        raise ValueError(f"Audio tensors differ from the MiniCPM-o 4.5 layout: {sorted(names.keys() ^ tensors.keys())}")

    projection = tensors["audio_projector.linear2.weight"]
    if len(projection.shape) != 2:
        raise ValueError(f"Invalid audio projector shape: {projection.shape}")

    writer.add_clip_has_audio_encoder(True)
    writer.add_clip_audio_projector_type("minicpmo4_5")
    writer.add_audio_embedding_length(get_field(reader, "d_model"))
    writer.add_audio_feed_forward_length(get_field(reader, "encoder_ffn_dim"))
    writer.add_audio_block_count(layers)
    writer.add_audio_projection_dim(int(projection.shape[1]))
    writer.add_audio_head_count(get_field(reader, "encoder_attention_heads"))
    writer.add_audio_attention_layernorm_eps(1e-5)
    writer.add_audio_num_mel_bins(get_field(reader, "n_mel"))
    writer.add_audio_projector_downsample_rate(get_field(reader, "audio_pool_step"))

    for source, target in names.items():
        tensor = tensors[source]
        writer.add_tensor(target, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)


def add_vision(writer: gguf.GGUFWriter, reader: gguf.GGUFReader) -> None:
    if get_field(reader, "clip.minicpmv_version") != 100045:
        raise ValueError("Expected the MiniCPM-o 4.5 vision encoder GGUF")

    writer.add_clip_vision_projector_type(get_field(reader, "clip.projector_type"))
    for key, field in reader.fields.items():
        if key.startswith("clip.") and key != "clip.projector_type":
            item_type = field.types[1] if len(field.types) > 1 else None
            writer.add_key_value(key, field.contents(), field.types[0], item_type)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="MiniCPM-o-4_5-audio-F16.gguf")
    parser.add_argument("--vision", type=Path, help="MiniCPM-o-4_5-vision-F16.gguf")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        output = Path(directory) / "mmproj.gguf"
        writer = gguf.GGUFWriter(output, "clip")
        try:
            writer.add_name("MiniCPM-o 4.5")
            if args.vision:
                add_vision(writer, gguf.GGUFReader(args.vision))
            add_audio(writer, gguf.GGUFReader(args.audio))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        try:
            args.output.hardlink_to(output)
        except FileExistsError:
            parser.error(f"Output already exists: {args.output}")


if __name__ == "__main__":
    main()
