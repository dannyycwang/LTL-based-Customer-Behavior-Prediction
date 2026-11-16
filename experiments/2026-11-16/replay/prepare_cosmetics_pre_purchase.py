"""Rebuild the seed-42 Act+Rel sessions up to, but excluding, first purchase."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import json
from zipfile import ZipFile

import pandas as pd

from audit_cosmetics_observation_boundary import ACTIONS, KEYS, ROOT, WORK, ZIP, sha
from prepare_cosmetics_seed42 import prepare_features
from verify_cosmetics_seed42 import notebook_sample


DATA = WORK / "cosmetics_pre_purchase_seed42"
AUDIT = ROOT / "cosmetics_pre_purchase_seed42_comparison"
SPLITS = ("train", "val", "test")
SEQ_COLS = ["behavior_sequence", "product_sequence", "category_sequence", "prop_BP"]
DTYPE = {name: str for name in [*KEYS, "session_id_hash", *SEQ_COLS]}
IDS = [*KEYS, "session_id_hash"]


def read_raw(selected):
    pieces = []
    columns = [*KEYS, "event_time", "event_type", "price", "product_id", "category_id"]
    raw_dtype = {key: str for key in [*KEYS, "product_id", "category_id"]}
    keys = selected[IDS]
    sessions = set(keys.user_session)
    with ZipFile(ZIP) as archive, archive.open("2019-Dec.csv") as stream:
        for chunk in pd.read_csv(stream, usecols=columns, dtype=raw_dtype, chunksize=250_000):
            chunk["row_id"] = chunk.index
            subset = chunk[chunk.user_session.isin(sessions)].merge(keys, on=KEYS, validate="many_to_one")
            subset["event_time"] = pd.to_datetime(subset.event_time.str.replace(" UTC", "", regex=False),
                                                   utc=True, errors="coerce")
            subset = subset.dropna(subset=["event_time", "event_type", *KEYS])
            subset = subset[pd.to_numeric(subset.price, errors="coerce").fillna(0) >= 0]
            pieces.append(subset.drop(columns=[*KEYS, "price"]))
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["session_id_hash", "event_time", "row_id"], kind="stable"
    )


def replay(events, original):
    event_types = set(events.event_type.unique())
    if event_types - {*ACTIONS, "purchase"}:
        raise ValueError(f"Unknown events: {event_types}")
    purchased = events.event_type.eq("purchase")
    positions = events.groupby("session_id_hash", sort=False).cumcount()
    first = positions[purchased].groupby(events.loc[purchased, "session_id_hash"]).min()
    raw_inputs = events.loc[~purchased].assign(
        action=lambda df: df.event_type.map(ACTIONS),
        product_id=lambda df: df.product_id.fillna("UNK"),
        category_id=lambda df: df.category_id.fillna("UNK"),
    )
    raw_replay = raw_inputs.groupby("session_id_hash", sort=False).agg(
        n_events=("action", "size"),
        behavior_sequence=("action", " ".join),
        product_sequence=("product_id", " ".join),
        category_sequence=("category_id", " ".join),
    ).reindex(original.index)
    raw_replay["full_n_events"] = events.groupby("session_id_hash").size()
    raw_replay["purchase"] = original.index.isin(first.index).astype(int)
    for field in raw_replay:
        matches = raw_replay[field].eq(original[field])
        if not matches.all():
            raise ValueError(f"{field}: {int((~matches).sum())} replay mismatches")
    before = first.reindex(original.index).fillna(original.n_events).astype(int)
    if (before > original.n_events).any():
        raise ValueError("Invalid first-purchase event position")
    return before


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    historical = ROOT / "cosmetics_seed42_audit"
    source = json.loads((historical / "manifest.json").read_text(encoding="utf-8"))
    boundary = json.loads((historical / "observation_boundary.json").read_text(encoding="utf-8"))
    archive_hash = sha(ZIP)
    if archive_hash != boundary["source_zip_sha256"]:
        raise ValueError("December archive changed since observation audit")
    frames = []
    for split in SPLITS:
        path = WORK / f"cosmetics_seed42_{split}.csv"
        if sha(path) != source["split_sha256"][split]:
            raise ValueError(f"Historical split hash mismatch: {split}")
        frames.append(pd.read_csv(path, dtype=DTYPE, usecols=[
            *IDS, "purchase", "n_events", "full_n_events", *SEQ_COLS
        ]).assign(split=split))
    original = pd.concat(frames, ignore_index=True).set_index("session_id_hash", verify_integrity=True)
    if original.duplicated(KEYS).any():
        raise ValueError("Duplicate raw user/session keys")
    events = read_raw(original.reset_index())
    before = replay(events, original)
    print(f"Raw action, product, category, label and event counts replayed for {len(original):,} sessions.",
          flush=True)
    removed = original.n_events - before
    if (removed[original.purchase == 0] != 0).any():
        raise ValueError("Found changed negative sessions")
    keep = before > 0
    comparison = original[["split", "purchase", "n_events"]].assign(
        pre_purchase_n=before, removed_after_first_purchase=removed, eligible=keep
    )
    comparison.to_csv(AUDIT / "observation_changes.csv.gz")
    formula_rows = []
    for split, seed in (("train", 42), ("val", 141)):
        sample = notebook_sample(original.loc[original.split == split].reset_index(), seed)
        sample_removed = removed.loc[sample.session_id_hash]
        formula_rows.append(dict(
            stage="fit" if split == "train" else "rank",
            n=len(sample),
            positive=int(sample.purchase.sum()),
            sessions_with_post_purchase_events=int(sample_removed.gt(0).sum()),
            zero_pre_purchase_events=int(before.loc[sample.session_id_hash].eq(0).sum()),
        ))
    pd.DataFrame(formula_rows).to_csv(AUDIT / "formula_basis_boundary.csv", index=False)
    paths, summary = {}, []
    for split in SPLITS:
        mask = original.split == split
        subset = original.loc[mask & keep, [*KEYS, "purchase", "n_events", *SEQ_COLS]].copy()
        truncated = removed.loc[subset.index] > 0
        for column in SEQ_COLS:
            subset.loc[truncated, column] = [
                " ".join(sequence.split()[:size])
                for sequence, size in zip(subset.loc[truncated, column], before.loc[subset.index[truncated]])
            ]
        subset["n_events"] = before.loc[subset.index]
        path = DATA / f"{split}.csv"
        subset.to_csv(path)
        paths[split] = path
        row = dict(split=split, historical_n=int(mask.sum()), pre_purchase_n=len(subset),
                   historical_positive=int(original.loc[mask, "purchase"].sum()),
                   pre_purchase_positive=int(subset.purchase.sum()),
                   zero_observed_excluded=int((mask & ~keep).sum()),
                   histories_truncated=int((mask & keep & removed.gt(0)).sum()),
                   events_removed=int(removed[mask].sum()),
                   sessions_longer_than_10=int(subset.n_events.gt(10).sum()),
                   sessions_longer_than_200=int(subset.n_events.gt(200).sum()),
                   maximum_length=int(subset.n_events.max()))
        summary.append(row)
    if summary[-1]["zero_observed_excluded"] != boundary["positives_with_zero_events_before_first_purchase"]:
        raise ValueError("Test empty-trace count differs from historical boundary audit")
    metadata = dict(seed=42, source_zip_sha256=archive_hash, verified_original_rows=len(original),
                    split_sha256={split: sha(path) for split, path in paths.items()},
                    historical_split_sha256=source["split_sha256"], basis_sha256=source["basis_sha256"],
                    positive_cutoff="Before first purchase in timestamp then source-row order",
                    negative_cutoff="All observed non-purchase events",
                    eligibility="At least one observed event; no resampling or split reassignment",
                    abstraction="Slice the original causal Act+Rel trace; propositions use only current and past event fields",
                    splits=summary)
    (AUDIT / "observation_boundary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(AUDIT / "observation_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    prepare_features({**source, "split_sha256": metadata["split_sha256"]}, paths, DATA / "features",
                     AUDIT / "feature_manifest.json",
                     "First-purchase cutoff with zero-observation sessions excluded")


if __name__ == "__main__":
    main()
