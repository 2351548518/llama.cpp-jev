# AMD XDNA

The XDNA backend uses XRT on Windows and Linux x86-64. It runs dense GGML `MUL_MAT`, including batched matrix multiplication with broadcasting. It accepts F32 activations and host weights in F32, F16, BF16, and GGUF formats that GGML can dequantize in rows, such as `Q4_K`, `Q6_K`, and `Q8_0`. It packs BF16 tiles for the NPU and accumulates the F32 partial results on the host when the reduction spans multiple tiles.

The scheduler sends attention, normalization, recurrent state, expert routing, and other unsupported operations to the CPU or another available backend. The backend keeps the model in its original host buffer format and caps staged activations at 4 MiB per loaded kernel. Eligible matrix multiplies have a reduction dimension of at least 256 and at least 128 output columns. Default dispatch starts at 256 input rows. This favors long prompts and vision batches and leaves single-token decoding on the CPU.

## Build

On Windows, use an x64 MSVC development shell, the AMD NPU driver, and the XRT SDK. On Linux, install the [AMD XDNA driver, firmware, and XRT shim](https://github.com/amd/xdna-driver) along with XRT development headers and libuuid development headers. The build searches the system paths and /opt/xilinx/xrt. Set `GGML_XDNA_XRT_ROOT` to the SDK directory when it is installed elsewhere.

```sh
cmake -S . -B build-xdna -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-xdna --target llama-server llama-bench test-backend-ops
```

Windows loads `xrt_coreutil.dll` from the AMD runtime. Linux needs `libxrt_coreutil.so` and the installed `amdxdna` plugin at runtime. Use `xrt-smi examine` to confirm the driver exposes the NPU before starting llama.cpp.

## Build kernels

Install [MLIR-AIE v1.4.1](https://github.com/Xilinx/mlir-aie/tree/v1.4.1) and its AIE compiler in an IRON environment. The [Windows setup guide](https://xilinx.github.io/mlir-aie/1.4.1/buildHostWinNative/) documents the native Windows toolchain. Run the kernel builder in either environment.

```sh
python ggml/src/ggml-xdna/build-kernels.py --mlir-aie path/to/mlir-aie --output xdna-kernel --target npu
```

Select `npu` for Phoenix or Hawk Point XDNA1 and `npu2` for XDNA2. Keep each target's files in its own directory. The required `matmul.xclbin` and `insts.bin` pair uses the original 256 x 1024 x 1024 layout. The optional pairs specialize low row counts, large reduction dimensions, and wide outputs. The backend loads each optional pair on demand and checks its layout before use.

Set `GGML_XDNA_KERNEL_DIR` to the generated directory before launching the tools. Set `GGML_XDNA_DEVICE_INDEX` if XRT exposes the NPU at an index other than 0. `GGML_XDNA_MIN_ROWS` overrides the batching threshold for profiling.

## Run

Select XDNA for text with `--device XDNA`. Use `--no-repack` to keep weights in standard GGUF layout. Keep `-ngl 0` so the model and KV cache remain in host memory.

```sh
build-xdna/bin/llama-server -m model.gguf --device XDNA -ngl 0 --no-repack
```

Select XDNA for vision with `--mmproj-device XDNA`. Combine both options to enable XDNA in the language and vision graphs. Add the `.exe` suffix to executable names on Windows.

```sh
build-xdna/bin/llama-server -m model.gguf --mmproj mmproj.gguf --device none -ngl 0 --mmproj-device XDNA
```

The device appears in `--list-devices` when the XRT device and the required kernel pair are available. Look for `first MUL_MAT executed on NPU` to confirm offload. A clean shutdown logs total operations, single-row operations, and NPU dispatches.

BF16 matrix multiplication can change token probabilities. The Gemma 3 270M Q4_K_M checkpoint passed the Windows XDNA1 generation checks with `-ub 256`. One prompt produced an early stop with `-ub 512`.

## Test and benchmark

Set `GGML_XDNA_MIN_ROWS=64` for the short-row regression cases. These tests compare `Q4_K`, `Q6_K`, `Q8_0`, F32, F16, and BF16 against the CPU and cover partial tiles, split reductions, and batch broadcasting.

```sh
build-xdna/bin/test-backend-ops test -b XDNA -o MUL_MAT -p '(m=1103,n=259,k=1024|m=403,n=97,k=1536|m=1153,n=259,k=3328|m=513,n=67,k=1280|m=4259,n=257,k=768),'
```

Restore the default batching threshold before comparing text performance. Run `llama-bench` with `--device XDNA` and repeat with `--device none` using the same model and settings.

```sh
build-xdna/bin/llama-bench -m model.gguf --device XDNA -ngl 0 --repack 0 -t 4 -p 128,512 -n 16 -r 3 -v
```
