from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ms_svbknet.config import load_config
from ms_svbknet.models import MSSVBKNet
from ms_svbknet.utils import write_json


class RestoreOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["restored"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="results/complexity.json")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
        device = torch.device(args.device)
    model = RestoreOnly(MSSVBKNet(cfg)).to(device).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    values = {
        "device": str(device),
        "input": [1, 3, args.size, args.size],
        "parameters": params,
        "parameters_millions": params / 1e6,
        "fp32_parameter_megabytes": params * 4 / 1e6,
    }
    sample = torch.randn(1, 3, args.size, args.size, device=device)
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(sample,), verbose=False)
        values["macs_giga"] = macs / 1e9
        values["flops_giga"] = 2 * macs / 1e9
    except ImportError:
        values["macs_giga"] = None
        values["flops_giga"] = None

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    latency = elapsed * 1000.0 / max(args.repeats, 1)
    values["latency_ms"] = latency
    values["fps"] = 1000.0 / latency
    values["peak_memory_megabytes"] = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    )
    write_json(values, Path(args.output))
    print(values)


if __name__ == "__main__":
    main()
