"""A CUDA probe for the dashboard's per-process GPU sampling.

    make session TASK=gpu-probe CMD="reference/.venv-rf/bin/python reference/gpu_probe.py"

Runs one 4096x4096 matrix product per iteration for fifteen seconds on the
first CUDA device, then prints the result's sum. `nvidia-smi` lists the process
while it runs, so the dashboard's GPU memory chart has a process line and the
GPU utilization chart has a real load. The default `reference/.venv` is CPU
only; `.venv-rf` is the CUDA torch.
"""
import time

import torch


def main():
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    y = x
    while time.time() - start < 15:
        y = x @ x
    torch.cuda.synchronize()
    print("gpu probe done", float(y.sum()))


if __name__ == "__main__":
    main()
