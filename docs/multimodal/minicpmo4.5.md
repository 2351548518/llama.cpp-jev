# MiniCPM-o 4.5

MiniCPM-o 4.5 accepts images, video frames and recorded audio and generates text through `llama-server` or `llama-mtmd-cli`. Download the language GGUF and the two F16 encoders from the [OpenBMB release](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf).

From the root of llama.cpp, combine the released vision and audio encoders into a libmtmd projector. Install the local `gguf` package with `python3 -m pip install ./gguf-py` if needed. The converter copies the tensor values and maps the Whisper weights to the libmtmd names.

```sh
python3 tools/mtmd/legacy-models/minicpmo45-convert-projectors.py \
    --audio /path/to/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf \
    --vision /path/to/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf \
    --output /path/to/MiniCPM-o-4_5-gguf/mmproj-F16.gguf

llama-server -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_0.gguf \
    --mmproj /path/to/MiniCPM-o-4_5-gguf/mmproj-F16.gguf \
    -c 8192 -b 1024 -ub 512 -ngl 999
```

Pass the published vision GGUF directly as `--mmproj` for image and video input, or omit `--vision` from the converter for an audio-only projector. Use `input_audio` in a `/v1/chat/completions` message for WAV input and `input_video` for video frames. You can send several media parts in a single message. FFmpeg and ffprobe are needed for video input.

The audio path handles a recorded clip per request. The reference [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) also has its own streaming session, speech decoder and vocoder for duplex conversation.
