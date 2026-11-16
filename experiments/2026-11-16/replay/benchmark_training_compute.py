"""Replay identical Coveo training minibatches for a controlled update-cost comparison."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmark_submission_runtime import (
    CONDITIONS, DEVICE, GROUPS, OUT, ROOT, encode_raw, encode_symbolic, load_model, source_for, trim_full_batch,
)
from report_coveo_seed42 import tex_table


def measure(model, loader, pos_weight, warmup):
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    iterator = iter(loader)
    model.train()

    def update(batch):
        args = [value.to(DEVICE, non_blocking=True) for value in batch]
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(*args[:-1]), args[-1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        return loss

    for _ in range(warmup):
        update(next(iterator))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for batch in iterator:
        loss = update(batch)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if not torch.isfinite(loss).item():
        raise ValueError("Nonfinite training update during timing")
    return seconds, torch.cuda.max_memory_allocated() / 2**20


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    source = source_for("Coveo", split="train")
    frame = source["test"]
    batch_size, warmup, measured = 256, 5, 40
    size = batch_size * (warmup + measured)
    positive = frame.purchase.mean()
    samples = np.random.default_rng(20261112).choice(len(frame), size=(3, size), replace=False)
    rows = []
    for repeat, indices in enumerate(samples, 1):
        selected = {**source, "test": frame.iloc[indices]}
        # Rotate condition order across repeats so each occupies each position once.
        choices = ("full_raw", "counts", "full_ltlf")
        for condition in (*choices[repeat - 1:], *choices[:repeat - 1]):
            run = ROOT / "runs" / f"{GROUPS['Coveo']}_{condition}"
            cfg = json.loads((run / "config.json").read_text())
            torch.manual_seed(42 + repeat)
            torch.cuda.manual_seed_all(42 + repeat)
            model, vocab, module = load_model("Coveo", condition, run, cfg)
            fields = encode_raw("Coveo", selected, cfg, vocab, module)
            if condition != "full_raw":
                fields["z"] = encode_symbolic(selected, condition)
            fields["y"] = selected["test"].purchase.to_numpy(np.float32)
            tensors = TensorDataset(*(torch.from_numpy(value) for value in fields.values()))
            loader = DataLoader(tensors, batch_size=batch_size, shuffle=False, pin_memory=True,
                                collate_fn=trim_full_batch if condition == "full_raw" else None)
            seconds, peak = measure(model, loader, (1 - positive) / positive, warmup)
            rows.append(dict(dataset="Coveo", condition=CONDITIONS[condition], repetition=repeat,
                             warmup_batches=warmup, measured_batches=measured, batch_size=batch_size,
                             measured_sec=seconds, milliseconds_per_update=1000 * seconds / measured,
                             peak_cuda_allocated_mib=peak))
            del model
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "coveo_update_timing_repetitions.csv", index=False)
    summary = frame.groupby("condition", sort=False).agg(
        median_ms_per_update=("milliseconds_per_update", "median"),
        min_ms_per_update=("milliseconds_per_update", "min"),
        max_ms_per_update=("milliseconds_per_update", "max"),
        peak_cuda_allocated_mib=("peak_cuda_allocated_mib", "max"),
    ).reset_index()
    full = summary.loc[summary.condition == "Full raw", "median_ms_per_update"].iloc[0]
    summary["speedup_vs_full_raw"] = full / summary.median_ms_per_update
    summary.to_csv(OUT / "coveo_update_timing.csv", index=False)
    tex_table(summary, OUT / "coveo_update_timing.tex", ".2f")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
