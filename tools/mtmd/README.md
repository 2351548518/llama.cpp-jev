# Multimodal Support in llama.cpp

This directory provides multimodal capabilities for `llama.cpp`. Initially intended as a showcase for running LLaVA models, its scope has expanded significantly over time to include various other vision-capable models. As a result, LLaVA is no longer the only multimodal architecture supported.

> [!IMPORTANT]
>
> Multimodal support can be viewed as a sub-project within `llama.cpp`. It is under **very heavy development**, and **breaking changes are expected**.

The naming and structure related to multimodal support have evolved, which might cause some confusion. Here's a brief timeline to clarify:

- [#3436](https://github.com/ggml-org/llama.cpp/pull/3436): Initial support for LLaVA 1.5 was added, introducing `llava.cpp` and `clip.cpp`. The `llava-cli` binary was created for model interaction.
- [#4954](https://github.com/ggml-org/llama.cpp/pull/4954): Support for MobileVLM was added, becoming the second vision model supported. This built upon the existing `llava.cpp`, `clip.cpp`, and `llava-cli` infrastructure.
- **Expansion & Fragmentation:** Many new models were subsequently added (e.g., [#7599](https://github.com/ggml-org/llama.cpp/pull/7599), [#10361](https://github.com/ggml-org/llama.cpp/pull/10361), [#12344](https://github.com/ggml-org/llama.cpp/pull/12344), and others). However, `llava-cli` lacked support for the increasingly complex chat templates required by these models. This led to the creation of model-specific binaries like `qwen2vl-cli`, `minicpmv-cli`, and `gemma3-cli`. While functional, this proliferation of command-line tools became confusing for users.
- [#12849](https://github.com/ggml-org/llama.cpp/pull/12849): `libmtmd` was introduced as a replacement for `llava.cpp`. Its goals include providing a single, unified command-line interface, improving the user/developer experience (UX/DX), and supporting both audio and image inputs.
- [#13012](https://github.com/ggml-org/llama.cpp/pull/13012): `mtmd-cli` was added, consolidating the various model-specific CLIs into a single tool powered by `libmtmd`.

## Pre-quantized models

See the list of pre-quantized model [here](../../docs/multimodal.md)

## How it works and what is `mmproj`?

Multimodal support in `llama.cpp` works by encoding images into embeddings using a separate model component, and then feeding these embeddings into the language model.

This approach keeps the multimodal components distinct from the core `libllama` library. Separating these allows for faster, independent development cycles. While many modern vision models are based on Vision Transformers (ViTs), their specific pre-processing and projection steps can vary significantly. Integrating this diverse complexity directly into `libllama` is currently challenging.

Consequently, running a multimodal model typically requires two GGUF files:
1.  The standard language model file.
2.  A corresponding **multimodal projector (`mmproj`)** file, which handles the image encoding and projection.

## What is `libmtmd`?

As outlined in the history, `libmtmd` is the modern library designed to replace the original `llava.cpp` implementation for handling multimodal inputs.

Originally built upon `clip.cpp` (similar to `llava.cpp`), `libmtmd` offers several advantages:
- **Unified Interface:** Aims to consolidate interaction for various multimodal models.
- **Improved UX/DX:** Features a more intuitive API, inspired by the `Processor` class in the Hugging Face `transformers` library.
- **Flexibility:** Designed to support multiple input types (text, audio, images) while respecting the wide variety of chat templates used by different models.

## How to obtain `mmproj`

Multimodal projector (`mmproj`) files are specific to each model architecture.

For the following models, you can use `convert_hf_to_gguf.py` with `--mmproj` flag to get the `mmproj` file:
- [Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) ; See the guide [here](../../docs/multimodal/gemma3.md) - Note: 1B variant does not have vision support
- SmolVLM (from [HuggingFaceTB](https://huggingface.co/HuggingFaceTB))
- SmolVLM2 (from [HuggingFaceTB](https://huggingface.co/HuggingFaceTB))
- [Pixtral 12B](https://huggingface.co/mistral-community/pixtral-12b) - only works with `transformers`-compatible checkpoint
- Qwen 2 VL and Qwen 2.5 VL (from [Qwen](https://huggingface.co/Qwen))
- [Mistral Small 3.1 24B](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
- InternVL 2.5 and InternVL 3 from [OpenGVLab](https://huggingface.co/OpenGVLab) (note: we don't support conversion of `InternVL3-*-hf` model, only non-HF version is supported ; `InternLM2Model` **text** model is not supported)
- [MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4_6) ; See the guide [here](../../docs/multimodal/minicpmv4.6.md) - requires the standard `transformers` v5.7.0+ checkpoint

For older models, please refer to the relevant guide for instructions on how to obtain or create them:

NOTE: conversion scripts are located under `tools/mtmd/legacy-models`

- [LLaVA](../../docs/multimodal/llava.md)
- [MobileVLM](../../docs/multimodal/MobileVLM.md)
- [GLM-Edge](../../docs/multimodal/glmedge.md)
- [MiniCPM-V 2.5](../../docs/multimodal/minicpmv2.5.md)
- [MiniCPM-V 2.6](../../docs/multimodal/minicpmv2.6.md)
- [MiniCPM-o 2.6](../../docs/multimodal/minicpmo2.6.md)
- [MiniCPM-V 4.0](../../docs/multimodal/minicpmv4.0.md)
- [MiniCPM-o 4.0](../../docs/multimodal/minicpmo4.0.md)
- [MiniCPM-o 4.5](../../docs/multimodal/minicpmo4.5.md)
- [MiniCPM-V 4.5](../../docs/multimodal/minicpmv4.5.md)
- [IBM Granite Vision](../../docs/multimodal/granitevision.md)

## Built-in video decoding

FFmpeg metadata inspection reads the caller's buffer during the call. A decoder that reads frames or audio keeps its own copy. The FFmpeg decoder caches up to four successful buffer probes per calling thread. Each key includes the SHA-256 of the input and the path to ffprobe. The cache stores metadata records. File-path inputs are probed on every load. Video stream duration takes precedence over container duration, including when an audio track is longer than the video.

## Native video decoding on Android and iOS

This fork builds `libmtmd` with shared-library ABI version 2. Rebuild native bindings and applications with the matching headers when upgrading from ABI version 1 or from upstream.

Applications can build with `-DLLAMA_SUBPROCESS=OFF -DMTMD_VIDEO=OFF` and provide `mtmd_helper_video_decoder` from `mtmd-helper.h`. This path does not launch ffmpeg or ffprobe. Use Android MediaCodec/ImageReader or iOS AVFoundation to implement the callbacks, then pass the same decoder to both metadata inspection and bitmap loading:

```cpp
mtmd_helper_init_opt options = mtmd_helper_init_opt_default();
options.video_params.decoder = &native_decoder;
options.video_params.center_frames = 4;
options.video_params.prompt_template = MTMD_VIDEO_TEMPLATE_QWEN3VL_WRAPPED;
options.video_params.no_soundtrack = true;

mtmd_helper_media_meta metadata;
if (mtmd_helper_get_media_meta(bytes, length, options, &metadata) != 0) {
    throw std::runtime_error("Unsupported or invalid media input");
}
auto media = mtmd_helper_bitmap_init_from_buf(mctx, bytes, length, false, options);
if (!media.bitmap) {
    throw std::runtime_error("Media decoding failed");
}
// Tokenize and consume media.bitmap while media.video_ctx is still alive.
mtmd_bitmap_free(media.bitmap);
if (media.video_ctx) {
    mtmd_helper_video_free(media.video_ctx);
}
```

The example options match the published four-frame Qwen3.5 decision checkpoint. Other checkpoints must use their own sampling/template settings. The callbacks must return display-oriented RGB24 at the requested width and height, with packed rows. `open` must own any bytes it retains; its input buffer is only valid during the call. Metadata inspection opens and closes its own decoder state without reading frames. Each subsequent load gets a fresh state. `close` must release that state's codec and buffer resources.

For exact source-frame selection, implement `start_selected`; its indices are sorted and may repeat. Return each distinct requested frame once, since the helper repeats it when needed. If it is not provided, `read_frame` must return every source frame in presentation order, including when asked to skip one with `rgb == NULL`. A random-keyframe seek does not preserve these sampling rules. Streams with variable frame rates or nonzero start timestamps need matching presentation-time handling in the application's decoder.

`test-mtmd-c-api` checks the callback lifetime and, with `MTMD_VIDEO=OFF`, decodes RGB pixels through the custom path. This is a portable integration check, not an Android/iPhone hardware performance measurement. Test the application adapter on its target device before reporting latency, memory or energy results.

For the optional four-frame video check, set `MTMD_TEST_MODEL` to a local Qwen3.5 model GGUF and `MTMD_TEST_MMPROJ` to its matching projector before running `test-mtmd-c-api`. The text model is opened vocabulary-only and the vision encoder is not executed; the check exercises real model-specific video configuration, sampling, RGB delivery and decoder cleanup without a GPU or ffmpeg.
