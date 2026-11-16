"""Train the five seed-42 Act+Rel conditions using the original dataset models."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd

from run_coveo import CONDITIONS, WORK, ROOT, sha
from verify_electronics_rebuild import DEFAULT_BASIS, DEFAULT_REBUILD

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DOWNLOADS = MODEL_DIR
SOURCES = {
    "electronics": DOWNLOADS / "electronics_ltlf_pipeline.py",
    "cosmetics": DOWNLOADS / "train_cosmetics_first_window_clean_names_v6_withBD_len20_fullfusion.py",
}
SPLITS = ("train", "val", "test")
PROTOCOL = "act_rel_seed42_v1"
PRE_PURCHASE_PROTOCOL = "pre_purchase_act_rel_seed42_v1"
PRE_PURCHASE_DATA = WORK / "cosmetics_pre_purchase_seed42"
PRE_PURCHASE_OUT = ROOT / "cosmetics_pre_purchase_seed42_comparison"
CAPS = {"electronics": 512, "cosmetics": 200}
DEFAULTS = {"electronics": (20, 4, 256, 256, "auprc"),
            "cosmetics": (25, 5, 512, 1024, "mcc")}


def original(dataset):
    path = SOURCES[dataset]
    spec = importlib.util.spec_from_file_location(f"original_{dataset}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(dataset, condition, module, observation="historical"):
    electronics = dataset == "electronics"
    if electronics:
        audit = ROOT / "electronics_rebuild" / "seed42_feature_manifest.json"
        data_dir = DEFAULT_REBUILD / "electronics_ready_noleak" / "electronics_splits" / "seed_42"
        paths = {key: data_dir / f"{key}.csv" for key in SPLITS}
        basis_path = DEFAULT_BASIS
        length_col = "session_len"
        raw_cols = ["behavior_sequence", "product_sku_hash_sequence", "category_hash_sequence"]
        master = DEFAULT_REBUILD / "electronics_ready_noleak" / "electronics_sessions_ltl_ready.csv"
    else:
        if observation == "pre_purchase":
            audit = PRE_PURCHASE_OUT / "feature_manifest.json"
            paths = {key: PRE_PURCHASE_DATA / f"{key}.csv" for key in SPLITS}
        else:
            audit = ROOT / "cosmetics_seed42_audit" / "feature_manifest.json"
            paths = {key: WORK / f"cosmetics_seed42_{key}.csv" for key in SPLITS}
        basis_path = COSMETICS_BASIS
        length_col = "n_events"
        raw_cols = ["behavior_sequence", "product_sequence", "category_sequence"]

    meta = json.loads(audit.read_text(encoding="utf-8"))
    if meta["seed"] != 42 or sha(basis_path) != meta["basis_sha256"]:
        raise ValueError("Saved seed-42 basis did not match audit")
    records = json.loads(basis_path.read_text(encoding="utf-8"))
    basis = (records if electronics else records["formulas"])[:10]
    frames = {}
    split_hashes = {}
    for key, path in paths.items():
        split_hashes[key] = sha(path)
        expected = meta["split_hashes"][f"42_{key}"] if electronics else meta["split_sha256"][key]
        if split_hashes[key] != expected:
            raise ValueError(f"Source split changed: {key}")
        frames[key] = pd.read_csv(path, usecols=["session_id_hash", "purchase", length_col, *raw_cols],
                                  dtype={"session_id_hash": str}).rename(columns={length_col: "length"})
    data = pd.concat(frames.values(), ignore_index=True)
    if not data.session_id_hash.is_unique or (data.length < 1).any():
        raise ValueError("Duplicate session IDs or empty session")
    cap = min(CAPS[dataset], int(data.length.max())) if condition == "full_raw" else 10
    length = np.minimum(data.length.to_numpy(), cap)
    index_by_split = np.split(np.arange(len(data)), np.cumsum([len(frame) for frame in frames.values()])[:-1])

    feature_hash = None
    z = None
    if condition in {"prefix_ltlf", "full_ltlf", "counts"}:
        if electronics:
            feature_path = resolve_saved_path(meta["feature_dir"]) / f"{condition}.npz"
            expected = meta["feature_hashes"][feature_path.name]
        else:
            feature = meta["features"][condition]
            feature_path, expected = resolve_saved_path(feature["path"]), feature["sha256"]
        feature_hash = sha(feature_path)
        if feature_hash != expected:
            raise ValueError("Saved feature file did not match audit")
        with np.load(feature_path) as npz:
            if not np.array_equal(npz["session_id_hash"], data.session_id_hash.to_numpy(str)):
                raise ValueError("Feature session order mismatch")
            if not np.array_equal(npz["y"], data.purchase.to_numpy()):
                raise ValueError("Feature label mismatch")
            z = npz["Z"].astype(np.float32)
            if condition != "counts" and npz["formulas"].tolist() != [row["formula_str"] for row in basis]:
                raise ValueError("Saved satisfaction basis mismatch")
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite features")

    parse = module.safe_load_list if electronics else module.parse_list_cell
    raw = [[parse(cell) for cell in data[column]] for column in raw_cols]
    for sequence in raw:
        if not np.array_equal([len(seq) for seq in sequence], data.length.to_numpy()):
            raise ValueError("Raw field length mismatch")
    vocab = None
    if electronics:
        fields = {}
        for name, sequence in zip(("action", "product", "category"), raw):
            packed = np.zeros((len(data), cap), dtype=np.int32)
            for row, (seq, size) in enumerate(zip(sequence, length)):
                values = [module.ACTION_TO_ID.get(str(value), 0) for value in seq[:size]] if name == "action" else seq[:size]
                packed[row, -size:] = values
            fields[name] = packed
        n_products, n_categories = module.get_vocab_sizes(master)
        vocab_sizes = dict(n_products=n_products, n_categories=n_categories)
    else:
        tokens = [[f"A={action}|P={product}|C={category}" for action, product, category in zip(*row)]
                  for row in zip(*raw)]
        vocab = {"<PAD>": 0, "<UNK>": 1}
        frequencies = Counter(token for seq in tokens[:len(frames["train"])] for token in seq)
        for token in sorted(frequencies, key=lambda value: (-frequencies[value], value)):
            vocab[token] = len(vocab)
        packed = np.zeros((len(data), cap), dtype=np.int32)
        for row, (seq, size) in enumerate(zip(tokens, length)):
            packed[row, :size] = [vocab.get(token, 1) for token in seq[:size]]
        fields = {"x": packed}
        vocab_sizes = {"vocab_size": len(vocab)}

    fields["length"] = length
    fields["y"] = data.purchase.to_numpy(np.float32)
    if z is not None:
        fields["z"] = z
    names = list(fields)
    tensors = [torch.from_numpy(field) for field in fields.values()]
    partitions = {key: TensorDataset(*(value[part[0]:part[-1] + 1] for value in tensors))
                  for key, part in zip(SPLITS, index_by_split)}
    return dict(frames=frames, names=names, partitions=partitions, cap=cap, basis=basis, vocab=vocab,
                vocab_sizes=vocab_sizes, z_dim=0 if z is None else z.shape[1], split_hashes=split_hashes,
                basis_sha256=meta["basis_sha256"], feature_sha256=feature_hash,
                feature_sec=meta["feature_construction_seconds_excluding_parsing"].get(condition, 0))


def model_for(dataset, module, data):
    cap = data["cap"]
    z_dim = data["z_dim"]
    if dataset == "electronics":
        model = module.FusionClassifier(feature_pack="apc_ltl" if z_dim else "apc_raw",
                                        z_dim=z_dim, max_len=cap, **data["vocab_sizes"])
        # Trimming all-padding columns preserves the historical right-aligned position IDs.
        model.raw_encoder.pos_emb.register_forward_pre_hook(
            lambda _embedding, args: (args[0] + cap - args[0].size(1),))
        return model
    return module.CosmeticsPredictor(mode="fusion" if z_dim else "raw", max_len=cap,
                                     z_dim=z_dim, **data["vocab_sizes"])


def batches(data, batch_size, eval_batch_size, device):
    return {key: DataLoader(ds, batch_size=batch_size if key == "train" else eval_batch_size,
                            shuffle=key == "train", pin_memory=device.type == "cuda")
            for key, ds in data["partitions"].items()}


def to_device(batch, names, dataset, device):
    fields = dict(zip(names, batch))
    lengths = fields.pop("length")
    width = int(lengths.max())
    steps = torch.arange(width)[None, :]
    if dataset == "electronics":
        for key in ("action", "product", "category"):
            fields[key] = fields[key][:, -width:]
        fields["mask"] = (steps >= width - lengths[:, None]).float()
    else:
        fields["x"] = fields["x"][:, :width]
        fields["mask"] = steps >= lengths[:, None]
    return {key: value.to(device, non_blocking=True) for key, value in fields.items()}


@torch.no_grad()
def predict(model, loader, data, dataset, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        inputs = to_device(batch, data["names"], dataset, device)
        labels.append(batch[data["names"].index("y")].numpy())
        probabilities.append(torch.sigmoid(model(inputs)).cpu().numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probabilities)


def metrics(module, y, p):
    if hasattr(module, "compute_metrics"):
        original_metrics = module.compute_metrics(y, p, threshold=0.5)
        result = {name.lower(): float(original_metrics[name])
                  for name in ("F1", "MCC", "AUPRC", "AUC", "Brier")}
    else:
        prediction = p >= 0.5
        result = {
            "f1": float(module.f1_score(y, prediction, zero_division=0)),
            "mcc": float(module.matthews_corrcoef(y, prediction)),
            "auprc": float(module.average_precision_score(y, p)),
            "auc": float(module.roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(module.brier_score_loss(y, p)),
        }
    if len(np.unique(y)) < 2:
        result.update(mcc=None, auprc=None, auc=None)
    return result


def update(model, batch, data, dataset, device, optimizer, criterion):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = to_device(batch, data["names"], dataset, device)
    loss = criterion(model(inputs), inputs["y"])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def atomic_json(path, content):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_checkpoint(path, content):
    tmp = path.with_suffix(".tmp")
    torch.save(content, tmp)
    tmp.replace(path)


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "random": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def benchmark(model, loaders, data, args, device, optimizer, criterion):
    torch.cuda.reset_peak_memory_stats(device)
    costs = []
    lengths = []
    for index, batch in enumerate(loaders["train"]):
        started = time.perf_counter()
        loss = update(model, batch, data, args.dataset, device, optimizer, criterion)
        if not math.isfinite(loss):
            raise ValueError("Nonfinite training loss in benchmark")
        torch.cuda.synchronize()
        if index >= 3:
            costs.append(time.perf_counter() - started)
            lengths.append(int(batch[data["names"].index("length")].max()))
        if len(costs) >= args.benchmark:
            break
    # Check the longest examples at the original batch sizes.
    for key, partition in data["partitions"].items():
        length = partition.tensors[data["names"].index("length")]
        batch_size = loaders[key].batch_size
        longest = int(length.argmax())
        indices = torch.arange(batch_size)
        indices[0] = longest
        batch = [values[indices] for values in partition.tensors]
        if key == "train":
            loss = update(model, batch, data, args.dataset, device, optimizer, criterion)
            if not math.isfinite(loss):
                raise ValueError("Nonfinite training loss in longest-session benchmark")
        else:
            model.eval()
            with torch.no_grad():
                logits = model(to_device(batch, data["names"], args.dataset, device))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits in longest-session benchmark")
    torch.cuda.synchronize()
    print(json.dumps(dict(dataset=args.dataset, condition=args.condition, timed_batches=len(costs),
                          mean_train_batch_sec=float(np.mean(costs)), max_train_width=max(lengths),
                          train_batches_per_epoch=len(loaders["train"]),
                          worst_case_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          parameters=sum(p.numel() for p in model.parameters()),
                          cap=data["cap"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--init-seed", type=int, help="Override downstream training RNG without changing the data split.")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--observation", choices=("historical", "pre_purchase"), default="historical")
    parser.add_argument("--benchmark", type=int, default=0, metavar="BATCHES")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if args.max_minutes is None and not (args.prepare_only or args.benchmark > 0):
        parser.error("Specify a positive --max-minutes for training.")
    if args.benchmark < 0 or (args.max_minutes is not None and args.max_minutes <= 0):
        parser.error("Invalid budget or benchmark size.")
    if args.observation == "pre_purchase" and args.dataset != "cosmetics":
        parser.error("The separate pre-purchase dataset applies to Cosmetics.")
    entered = time.perf_counter()
    deadline = math.inf if args.max_minutes is None else entered + 60 * args.max_minutes
    protocol = PRE_PURCHASE_PROTOCOL if args.observation == "pre_purchase" else PROTOCOL
    init_seed = args.seed if args.init_seed is None else args.init_seed
    suffix = "" if args.init_seed is None else f"_init{init_seed}"
    run = args.out / f"{args.dataset}_{protocol}_{args.condition}{suffix}"
    module = original(args.dataset)
    if (run / "metrics.json").exists() and not (args.prepare_only or args.benchmark):
        print("Already complete:", run, flush=True)
        return
    data = inputs(args.dataset, args.condition, module, args.observation)
    epochs, patience, batch_size, eval_batch_size, monitor = DEFAULTS[args.dataset]
    print(f"{args.dataset} {args.condition} prepared {data['vocab_sizes']} cap={data['cap']}"
          f" z={data['z_dim']} in {time.perf_counter() - entered:.1f}s", flush=True)
    if args.prepare_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    module.set_seed(init_seed)
    loaders = batches(data, batch_size, eval_batch_size, device)
    model = model_for(args.dataset, module, data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_y = data["frames"]["train"].purchase
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train_y) - train_y.sum()) / train_y.sum()],
                                                            device=device, dtype=torch.float32))
    if args.benchmark:
        benchmark(model, loaders, data, args, device, optimizer, criterion)
        return

    config = dict(dataset=args.dataset, seed=args.seed, condition=args.condition, protocol=protocol,
                  split_source_sha256=data["split_hashes"], basis_sha256=data["basis_sha256"],
                  feature_sha256=data["feature_sha256"], source_sha256=sha(SOURCES[args.dataset]),
                  runner_sha256=sha(Path(__file__)), raw_window="first", raw_len=data["cap"],
                  feature_dim=data["z_dim"], epochs=epochs, patience=patience, batch_size=batch_size,
                  eval_batch_size=eval_batch_size, monitor=monitor, threshold=0.5,
                  vocab_sizes=data["vocab_sizes"], gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda)
    if args.init_seed is not None:
        config["init_seed"] = init_seed
    run.mkdir(parents=True, exist_ok=True)
    cfg_path = run / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Run config changed; use a different output directory")
    atomic_json(cfg_path, config)
    atomic_json(run / "basis.json", data["basis"])
    if data["vocab"] is not None:
        atomic_json(run / "vocab.json", data["vocab"])
    for key, frame in data["frames"].items():
        frame[["session_id_hash", "purchase", "length"]].rename(
            columns={"session_id_hash": "session_id", "purchase": "label"}).to_csv(run / f"{key}_ids.csv", index=False)

    ck_path = run / "last_checkpoint.pt"
    last_epoch, best_epoch, bad, elapsed_prior = 0, 0, 0, 0.0
    best_score = -math.inf
    best_model = None
    history = []
    if ck_path.exists():
        state = torch.load(ck_path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Checkpoint config changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        last_epoch, best_epoch, bad = state["epoch"], state["best_epoch"], state["bad"]
        elapsed_prior, best_score = state["elapsed"], state["best_score"]
        best_model, history = state["best_model"], state["history"]
        restore_rng(state["rng"])
    torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(last_epoch + 1, epochs + 1):
        if bad >= patience:
            break
        losses = []
        for batch in loaders["train"]:
            if time.perf_counter() >= deadline:
                print("Budget reached; last completed epoch retained.", flush=True)
                return
            loss = update(model, batch, data, args.dataset, device, optimizer, criterion)
            if not math.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            losses.append(loss)
        y, p = predict(model, loaders["val"], data, args.dataset, device)
        score = metrics(module, y, p)[monitor]
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
        last_epoch = epoch
        torch.cuda.synchronize()
        elapsed = elapsed_prior + time.perf_counter() - started
        row = dict(epoch=epoch, monitor=monitor, score=score, loss=float(np.mean(losses)), elapsed_sec=elapsed)
        history.append(row)
        atomic_checkpoint(ck_path, dict(config=config, epoch=epoch, best_epoch=best_epoch,
                                        best_score=best_score, bad=bad, elapsed=elapsed, history=history,
                                        model=model.state_dict(), optimizer=optimizer.state_dict(),
                                        best_model=best_model, rng=rng_state()))
        atomic_json(run / "epochs.json", history)
        print(json.dumps(row), flush=True)

    train_sec = elapsed_prior + time.perf_counter() - started
    if time.perf_counter() >= deadline:
        print("Budget reached before test; trained checkpoint retained.", flush=True)
        return
    model.load_state_dict(best_model)
    torch.cuda.synchronize()
    inference = time.perf_counter()
    y, p = predict(model, loaders["test"], data, args.dataset, device)
    torch.cuda.synchronize()
    infer_sec = time.perf_counter() - inference
    test = data["frames"]["test"]
    if not np.array_equal(y, test.purchase.to_numpy()) or not np.isfinite(p).all():
        raise ValueError("Test labels misaligned or invalid probabilities")
    pred = pd.DataFrame(dict(session_id=test.session_id_hash, label=y, probability=p,
                             length=test.length, threshold=0.5, seed=args.seed))
    pred.to_csv(run / "predictions.csv", index=False)
    groups = {"all": np.ones(len(test), bool), "L<=10": test.length <= 10, "L>10": test.length > 10}
    rows = [dict(stratum=key, n=int(mask.sum()), positive_rate=float(y[mask].mean()),
                 **metrics(module, y[mask], p[mask])) for key, mask in groups.items()]
    pd.DataFrame(rows).to_csv(run / "stratified_metrics.csv", index=False)
    atomic_checkpoint(run / "best_model.pt", best_model)
    result = dict(**metrics(module, y, p), dataset=args.dataset, seed=args.seed, condition=args.condition,
                  protocol=protocol, threshold=0.5, best_epoch=best_epoch, actual_epochs_ran=last_epoch,
                  n_params=sum(value.numel() for value in model.parameters()), feature_dim=data["z_dim"],
                  train_including_validation_sec=train_sec, inference_precomputed_features_sec=infer_sec,
                  feature_construction_all_splits_sec=data["feature_sec"], gpu=config["gpu"])
    if args.init_seed is not None:
        result["init_seed"] = init_seed
    atomic_json(run / "metrics.json", result)
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
