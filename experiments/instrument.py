# Copyright (c) 2026 RoboJEPA experiment contributors.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Run the upstream policy client with scoped, frozen-model instrumentation."""
import atexit
from contextlib import nullcontext
import os
from pathlib import Path
import random
import runpy
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .runner import ROOT, write_json


class Recorder:
    def __init__(self, count_flops=True):
        self.count_flops = count_flops
        self.calls = []
        self.decisions = []
        self.decision = None

    def wrap(self, function, module):
        def measured(*args, **kwargs):
            tensor = args[0]
            batch = int(tensor.shape[0])
            cuda = tensor.is_cuda
            if cuda:
                torch.cuda.synchronize(tensor.device)
                before = torch.cuda.memory_allocated(tensor.device)
                torch.cuda.reset_peak_memory_stats(tensor.device)
            else:
                before = 0
            counter = FlopCounterMode(display=False) if self.count_flops else None
            start = time.perf_counter()
            with counter if counter is not None else nullcontext():
                output = function(*args, **kwargs)
            if cuda:
                torch.cuda.synchronize(tensor.device)
            elapsed = time.perf_counter() - start
            features = output[0] if isinstance(output, tuple) else output
            # Count visual embeddings, excluding channel dimension. Predictor output
            # can include context positions: these are computed output tokens too.
            tokens = features.numel() // features.shape[-1]
            after = torch.cuda.memory_allocated(tensor.device) if cuda else 0
            peak = torch.cuda.max_memory_allocated(tensor.device) if cuda else 0
            self.calls.append({
                "module": module, "decision": self.decision, "batch_size": batch,
                "output_tokens": tokens, "latency_s": elapsed,
                "amortized_sample_latency_s": elapsed / batch,
                "flops": counter.get_total_flops() if counter is not None else None,
                "allocated_before_bytes": before if cuda else None,
                "allocated_after_bytes": after if cuda else None,
                "peak_allocated_bytes": peak if cuda else None,
                "peak_increment_bytes": max(0, peak - before) if cuda else None,
            })
            return output
        return measured

    def dump(self, path):
        write_json(path, {"schema_version": 1, "count_flops": self.count_flops,
                          "calls": self.calls, "decisions": self.decisions})


def main():
    from app.robo_jepa.mpc import mpc_utils
    from app.robo_jepa.world_model import WorldModel

    trial = Path(os.environ["EXPERIMENT_TRIAL"])
    seed = int(os.environ["EXPERIMENT_SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_grad_enabled(False)
    recorder = Recorder(os.environ.get("EXPERIMENT_COUNT_FLOPS", "1") == "1")
    stats = trial / "efficiency.json"
    atexit.register(recorder.dump, stats)
    original_load = WorldModel.load_pretrained.__func__
    original_cem = mpc_utils.cem

    def load(cls, *args, **kwargs):
        wm = original_load(cls, *args, **kwargs)
        for module in (wm.transition_model, wm.encoder):
            if module is not None:
                module.eval()
                module.requires_grad_(False)
        wm.forward = recorder.wrap(wm.forward, "predictor")
        if wm.encoder is not None:
            wm.encoder.forward = recorder.wrap(wm.encoder.forward, "encoder")
        write_json(trial / "hardware.json", {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "seed": seed,
            "predictor_parameters": sum(p.numel() for p in wm.transition_model.parameters()),
            "encoder_parameters": sum(p.numel() for p in wm.encoder.parameters()),
            "trainable_parameters": sum(p.numel() for m in (wm.transition_model, wm.encoder) for p in m.parameters() if p.requires_grad),
        })
        return wm

    def cem(*args, **kwargs):
        if not recorder.decisions:
            # Preserve the original sweep's paired CEM seed, independent of
            # random numbers consumed while initializing different model sizes.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        recorder.decision = len(recorder.decisions) + 1
        first = len(recorder.calls)
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            result = original_cem(*args, **kwargs)
            torch.cuda.synchronize()
            recorder.decisions.append({"index": recorder.decision, "global_step": kwargs.get("global_step"),
                                       "latency_s": time.perf_counter() - start,
                                       "first_call": first, "last_call_exclusive": len(recorder.calls)})
            return result
        finally:
            recorder.decision = None
            recorder.dump(stats)

    WorldModel.load_pretrained = classmethod(load)
    mpc_utils.cem = cem
    client = ROOT / "scripts/deployment/agents/world_model/policy_client.py"
    sys.path.insert(0, str(client.parent))
    sys.argv = [str(client)] + sys.argv[1:]
    try:
        runpy.run_path(str(client), run_name="__main__")
    finally:
        recorder.dump(stats)


if __name__ == "__main__":
    main()
