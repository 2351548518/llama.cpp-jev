#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Build GGML XDNA kernels with MLIR-AIE v1.4.1.")
    parser.add_argument("--mlir-aie", required=True, type=Path, help="Path to the MLIR-AIE source tree.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for the xclbin and instruction files.")
    parser.add_argument("--target", required=True, choices=["npu", "npu2"], help="XDNA generation to compile for.")
    args = parser.parse_args()

    example = args.mlir_aie.resolve() / "programming_examples/basic/matrix_multiplication/whole_array/whole_array.py"
    if not example.is_file():
        parser.error(f"MLIR-AIE example not found at {example}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for suffix, m, k, n, core_m in (
        ("", 256, 1024, 1024, 32),
        ("-small", 128, 1024, 1024, 16),
        ("-wide", 256, 4096, 1024, 32),
        ("-output", 256, 1024, 4096, 32),
    ):
        subprocess.run([
            sys.executable, str(example), "--dev", args.target,
            "-M", str(m), "-K", str(k), "-N", str(n),
            "-m", str(core_m), "-k", "32", "-n", "32", "--n-aie-cols", "4",
            "--b-col-maj", "1", "--c-col-maj", "0", "--dtype_in", "bf16", "--dtype_out", "f32",
            f"--xclbin-path={output / ('matmul' + suffix + '.xclbin')}",
            f"--insts-path={output / ('insts' + suffix + '.bin')}",
        ], cwd=output, check=True)


if __name__ == "__main__":
    main()
