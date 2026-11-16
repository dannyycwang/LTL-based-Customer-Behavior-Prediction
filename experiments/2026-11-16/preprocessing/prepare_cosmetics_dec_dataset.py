#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_cosmetics_dec_dataset.py

Prepare the Cosmetics Shop clickstream dataset for LTLf full-session compression experiments.

Protocol:
- Dataset: Kaggle mkechinov/ecommerce-events-history-in-cosmetics-shop, e.g. 2019-Dec.csv
- Session label: whether the full session contains a purchase event.
- Input traces: purchase events are removed from raw/LTLf input traces.
- Cosmetics event mapping:
    view             -> d  (product detail)
    cart             -> a  (add to cart)
    remove_from_cart -> r  (remove from cart)
    purchase         -> label only, removed from input trace
- Dwell:
    train-only log1p tertiles after 99% cap, using seed 42 train split.
    dl/dm/dh
- Product/category symbolic propositions:
    hp/np/sp/pc/fp/rp/p2/p3/dp1/dp2/dp3
    hc/sc/cc/rc
- Formula learning later should use last 10 events.
- Raw/fusion training later should use first 10 events.
- Full-session model checking later should use full input trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Constants
# -----------------------------

DEFAULT_SEEDS = [42, 43, 44, 45]
LABEL_COL = "purchase"
GROUP_COL = "user_id"

CODEBOOK = {
    "b": "pageview / browse",
    "d": "product detail",
    "a": "add to cart",
    "r": "remove from cart",
    "dl": "dwell low",
    "dm": "dwell medium",
    "dh": "dwell high",
    "hp": "has product",
    "np": "no product",
    "sp": "same product as previous product-bearing event",
    "pc": "product changed from previous product-bearing event",
    "fp": "first time product in session",
    "rp": "revisit seen product",
    "p2": "product seen at least twice",
    "p3": "product seen at least three times",
    "dp1": "one distinct product seen so far",
    "dp2": "two distinct products seen so far",
    "dp3": "three or more distinct products seen so far",
    "hc": "has category",
    "sc": "same category as previous product-bearing event",
    "cc": "category changed",
    "rc": "revisit seen category",
}

VOCAB_ORDER = {
    "b": 0, "d": 1, "a": 2, "r": 3,
    "dl": 4, "dm": 5, "dh": 6,
    "hp": 7, "np": 8, "sp": 9, "pc": 10,
    "fp": 11, "rp": 12, "p2": 13, "p3": 14,
    "dp1": 15, "dp2": 16, "dp3": 17,
    "hc": 18, "sc": 19, "cc": 20, "rc": 21,
}

BEHAVIOR_MAP_INPUT = {
    "view": "d",
    "cart": "a",
    "remove_from_cart": "r",
}

BEHAVIOR_ID_MAP_INPUT = {
    "view": 2,
    "cart": 3,
    "remove_from_cart": 4,
}

EXPECTED_EVENTS = {"view", "cart", "remove_from_cart", "purchase"}

USECOLS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

DTYPE_MAP = {
    "event_type": "string",
    "product_id": "string",
    "category_id": "string",
    "category_code": "string",
    "brand": "string",
    "user_id": "string",
    "user_session": "string",
}


# -----------------------------
# Utility functions
# -----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def make_session_hash(user_id: object, user_session: object) -> str:
    s = f"{user_id}||{user_session}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def join_tokens(x: Iterable[object]) -> str:
    return " ".join(map(str, x))


def join_float_tokens(x: Iterable[object]) -> str:
    return " ".join(f"{float(v):.3f}" for v in x)


def alpha_symbol(n: int) -> str:
    """0 -> a, 25 -> z, 26 -> aa, ..."""
    letters: List[str] = []
    while True:
        letters.append(chr(ord("a") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


def build_symbol_map(series: pd.Series) -> Dict[str, str]:
    vals = series.value_counts().index.tolist()
    return {str(v): alpha_symbol(i) for i, v in enumerate(vals)}


def combine_prop_strings(*cols: Sequence[object]) -> List[str]:
    out: List[str] = []
    for vals in zip(*cols):
        props: List[str] = []
        for v in vals:
            if pd.isna(v) or str(v) == "":
                continue
            props.extend(str(v).split("+"))

        props = sorted(set(props), key=lambda x: VOCAB_ORDER.get(x, 999))
        out.append("+".join(props))
    return out


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required. In Colab run: !pip -q install scikit-learn"
        ) from exc


def download_kaggle_dataset(target_file: str) -> Path:
    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError(
            "kagglehub is required for --download_kaggle. "
            "In Colab run: !pip -q install kagglehub"
        ) from exc

    dataset_path = Path(
        kagglehub.dataset_download("mkechinov/ecommerce-events-history-in-cosmetics-shop")
    )
    csv_path = dataset_path / target_file

    if not csv_path.exists():
        available = [p.name for p in sorted(dataset_path.glob("*"))]
        raise FileNotFoundError(
            f"Cannot find {target_file} in Kaggle dataset path {dataset_path}. "
            f"Available files: {available}"
        )

    return csv_path


# -----------------------------
# Loading and cleaning
# -----------------------------

def load_cosmetics_csv(csv_path: Path, nrows: int | None = None) -> pd.DataFrame:
    log(f"[1/12] Loading CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPE_MAP,
        low_memory=False,
        nrows=nrows,
    )

    log(f"Raw shape: {df.shape}")
    log(f"Columns: {df.columns.tolist()}")

    df["row_id"] = np.arange(len(df), dtype=np.int64)

    # Parse time. Kaggle often has "UTC" suffix.
    df["event_time"] = (
        df["event_time"]
        .astype(str)
        .str.replace(" UTC", "", regex=False)
    )
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["event_time", "event_type", "user_id", "user_session"]).copy()
    df = df[df["price"].fillna(0) >= 0].copy()
    after = len(df)

    log(f"Rows after cleaning: {after} (dropped {before - after})")
    log("Event type counts:")
    log(str(df["event_type"].value_counts(dropna=False)))

    unknown_events = sorted(set(df["event_type"].dropna().unique()) - EXPECTED_EVENTS)
    log(f"Unknown event types: {unknown_events}")
    if unknown_events:
        raise ValueError(f"Unexpected event_type values: {unknown_events}")

    df["session_id_hash"] = [
        make_session_hash(u, s)
        for u, s in zip(df["user_id"].astype(str), df["user_session"].astype(str))
    ]

    df = df.sort_values(
        ["session_id_hash", "event_time", "row_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    log(f"Unique sessions: {df['session_id_hash'].nunique()}")
    log(f"Unique users: {df['user_id'].nunique()}")

    return df


# -----------------------------
# Labels and input traces
# -----------------------------

def build_labels_and_input(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("[2/12] Building session labels and removing purchase from input traces")

    df = df.copy()
    df["purchase_event"] = (df["event_type"] == "purchase").astype("int8")

    session_label = (
        df.groupby("session_id_hash", sort=False)
        .agg(
            user_id=("user_id", "first"),
            user_session=("user_session", "first"),
            full_start_time=("event_time", "min"),
            full_end_time=("event_time", "max"),
            full_n_events=("event_type", "size"),
            purchase=("purchase_event", "max"),
        )
        .reset_index()
    )

    log(f"All full-log sessions: {len(session_label)}")
    log(f"Full-log purchase session ratio: {session_label['purchase'].mean():.6f}")
    log(str(session_label["purchase"].value_counts()))

    # Remove purchase event from all input traces.
    df_in = df[df["event_type"] != "purchase"].copy()

    df_in["behavior_token"] = df_in["event_type"].map(BEHAVIOR_MAP_INPUT).astype("string")
    df_in["behavior_id"] = df_in["event_type"].map(BEHAVIOR_ID_MAP_INPUT).astype("int16")

    if df_in["behavior_token"].isna().any():
        bad = df_in[df_in["behavior_token"].isna()]["event_type"].value_counts()
        raise ValueError(f"NaN behavior token after mapping: {bad}")

    if "purchase" in set(df_in["event_type"]):
        raise AssertionError("purchase should not appear in df_in")

    log(f"Input events after removing purchase: {len(df_in)}")
    log("Input event type counts:")
    log(str(df_in["event_type"].value_counts()))

    # Drop sessions with no input trace.
    input_session_ids = set(df_in["session_id_hash"].astype(str))
    before_sessions = len(session_label)

    session_label = session_label[
        session_label["session_id_hash"].astype(str).isin(input_session_ids)
    ].copy()

    after_sessions = len(session_label)

    log(f"Sessions before dropping empty input traces: {before_sessions}")
    log(f"Sessions after dropping empty input traces : {after_sessions}")
    log(f"Dropped sessions: {before_sessions - after_sessions}")
    log(f"Purchase ratio after drop: {session_label['purchase'].mean():.6f}")

    return session_label, df_in


# -----------------------------
# Preliminary sessions and splits
# -----------------------------

def build_preliminary_sessions(
    df_in: pd.DataFrame,
    session_label: pd.DataFrame,
    prep: Path,
) -> pd.DataFrame:
    log("[3/12] Building preliminary session table")

    session_base = (
        df_in.groupby("session_id_hash", sort=False)
        .agg(
            user_id=("user_id", "first"),
            user_session=("user_session", "first"),
            start_time=("event_time", "min"),
            end_time=("event_time", "max"),
            n_events=("event_type", "size"),
        )
        .reset_index()
    )

    session_base = session_base.merge(
        session_label[
            [
                "session_id_hash",
                "purchase",
                "full_n_events",
                "full_start_time",
                "full_end_time",
            ]
        ],
        on="session_id_hash",
        how="inner",
    )

    session_base["duration_sec"] = (
        session_base["end_time"] - session_base["start_time"]
    ).dt.total_seconds().fillna(0)

    log(f"Session base shape: {session_base.shape}")
    log(f"Purchase ratio: {session_base['purchase'].mean():.6f}")
    log("Length summary:")
    log(str(session_base["n_events"].describe(percentiles=[.5, .75, .9, .95, .99])))

    out = prep / "cosmetics_dec_sessions_preliminary.csv"
    session_base.to_csv(out, index=False)
    log(f"Saved: {out}")

    return session_base


def sample_negatives(split_df: pd.DataFrame, seed: int, neg_ratio: int = 4) -> pd.DataFrame:
    pos = split_df[split_df[LABEL_COL] == 1].copy()
    neg = split_df[split_df[LABEL_COL] == 0].copy()

    n_pos = len(pos)
    if n_pos == 0:
        raise ValueError("No positives in split. Need to adjust split strategy.")

    n_neg_target = min(len(neg), neg_ratio * n_pos)
    if len(neg) > n_neg_target:
        neg_s = neg.sample(n=n_neg_target, random_state=seed)
    else:
        neg_s = neg

    out = pd.concat([pos, neg_s], axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def make_seed_split(sessions: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_sklearn()
    from sklearn.model_selection import StratifiedGroupKFold

    sessions = sessions.reset_index(drop=True)
    y = sessions[LABEL_COL].values
    groups = sessions[GROUP_COL].astype(str).values

    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf1.split(sessions, y, groups))

    train_val = sessions.iloc[train_val_idx].reset_index(drop=True)
    test = sessions.iloc[test_idx].reset_index(drop=True)

    y_tv = train_val[LABEL_COL].values
    groups_tv = train_val[GROUP_COL].astype(str).values

    sgkf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1000)
    train_idx, val_idx = next(sgkf2.split(train_val, y_tv, groups_tv))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    users_train = set(train[GROUP_COL].astype(str))
    users_val = set(val[GROUP_COL].astype(str))
    users_test = set(test[GROUP_COL].astype(str))

    assert users_train.isdisjoint(users_val)
    assert users_train.isdisjoint(users_test)
    assert users_val.isdisjoint(users_test)

    return train, val, test


def build_splits(
    session_base: pd.DataFrame,
    split_dir: Path,
    seeds: Sequence[int],
    neg_ratio: int,
) -> None:
    log("[4/12] Building user-level seed splits")

    split_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_dir = split_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        train, val, test = make_seed_split(session_base, seed)

        train_s = sample_negatives(train, seed=seed, neg_ratio=neg_ratio)
        val_s = sample_negatives(val, seed=seed, neg_ratio=neg_ratio)
        test_s = sample_negatives(test, seed=seed, neg_ratio=neg_ratio)

        train_s.to_csv(seed_dir / "train.csv", index=False)
        val_s.to_csv(seed_dir / "val.csv", index=False)
        test_s.to_csv(seed_dir / "test.csv", index=False)

        log(f"\nSeed {seed}")
        for name, split in [("train", train_s), ("val", val_s), ("test", test_s)]:
            log(
                f"{name}: shape={split.shape}, "
                f"pos={int(split[LABEL_COL].sum())}, "
                f"neg={int((split[LABEL_COL] == 0).sum())}, "
                f"pos_rate={split[LABEL_COL].mean():.4f}, "
                f"users={split[GROUP_COL].nunique()}, "
                f"sessions={split['session_id_hash'].nunique()}"
            )


# -----------------------------
# Dwell fitting
# -----------------------------

def add_dwell_tokens(
    df_in: pd.DataFrame,
    split_dir: Path,
    prep: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    log("[5/12] Computing dwell time and fitting train-only dwell buckets")

    df_in = df_in.sort_values(
        ["session_id_hash", "event_time", "row_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    df_in["next_event_time"] = df_in.groupby("session_id_hash")["event_time"].shift(-1)
    df_in["dwell_sec"] = (
        (df_in["next_event_time"] - df_in["event_time"])
        .dt.total_seconds()
        .fillna(0)
        .clip(lower=0)
    )

    log("Dwell summary before capping:")
    log(str(df_in["dwell_sec"].describe(percentiles=[.5, .75, .9, .95, .99, .995, .999])))

    train42_path = split_dir / "seed_42" / "train.csv"
    if not train42_path.exists():
        raise FileNotFoundError(f"Need seed_42 split before fitting dwell: {train42_path}")

    train42 = pd.read_csv(train42_path, dtype={"session_id_hash": str})
    train42_ids = set(train42["session_id_hash"].astype(str))

    train_dwell = df_in.loc[
        df_in["session_id_hash"].astype(str).isin(train42_ids),
        "dwell_sec",
    ].copy()

    cap_seconds = float(train_dwell.quantile(0.99))
    train_dwell_capped = train_dwell.clip(upper=cap_seconds)
    train_log = np.log1p(train_dwell_capped)

    q33_log, q66_log = train_log.quantile([1 / 3, 2 / 3]).values

    def dwell_to_token(x: object) -> str:
        x_float = min(max(float(x), 0.0), cap_seconds)
        lx = np.log1p(x_float)

        if lx <= q33_log:
            return "dl"
        if lx <= q66_log:
            return "dm"
        return "dh"

    df_in["dwell_token"] = df_in["dwell_sec"].map(dwell_to_token).astype("string")

    log("Dwell token counts:")
    log(str(df_in["dwell_token"].value_counts().sort_index()))

    dwell_stats: Dict[str, object] = {
        "method": "train_only_log1p_tertiles_after_99pct_cap",
        "cap_seconds": cap_seconds,
        "median_seconds": float(train_dwell.median()),
        "q33_log": float(q33_log),
        "q66_log": float(q66_log),
        "split_seed_for_boundaries": 42,
    }

    out = prep / "cosmetics_dec_dwell_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dwell_stats, f, indent=2)

    log(f"Saved: {out}")

    return df_in, dwell_stats


# -----------------------------
# Product/category propositions
# -----------------------------

def add_product_category_props(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    log("[6/12] Building product/category symbolic propositions")

    df_in = df_in.copy()

    df_in["product_raw"] = df_in["product_id"].fillna("UNK").astype(str)

    cat_code_coverage = float(df_in["category_code"].notna().mean())
    brand_coverage = float(df_in["brand"].notna().mean()) if "brand" in df_in.columns else 0.0

    category_source = "category_code" if cat_code_coverage >= 0.70 else "category_id"

    df_in["category_raw"] = df_in[category_source].fillna("UNK").astype(str)
    if "brand" in df_in.columns:
        df_in["brand_raw"] = df_in["brand"].fillna("UNK").astype(str)
    else:
        df_in["brand_raw"] = "UNK"

    log(f"CATEGORY_SOURCE: {category_source}")
    log(f"category_code coverage: {cat_code_coverage:.6f}")
    log(f"brand coverage: {brand_coverage:.6f}")
    log(f"product unique: {df_in['product_raw'].nunique()}")
    log(f"category unique: {df_in['category_raw'].nunique()}")

    df_in = df_in.sort_values(
        ["session_id_hash", "event_time", "row_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    df_in["has_product"] = df_in["product_raw"].ne("UNK")
    df_in["has_category"] = df_in["category_raw"].ne("UNK")

    # Previous product/category-bearing event inside session.
    df_in["prev_product_bearing"] = (
        df_in["product_raw"]
        .where(df_in["has_product"])
        .groupby(df_in["session_id_hash"])
        .ffill()
        .groupby(df_in["session_id_hash"])
        .shift(1)
    )

    df_in["prev_category_bearing"] = (
        df_in["category_raw"]
        .where(df_in["has_category"])
        .groupby(df_in["session_id_hash"])
        .ffill()
        .groupby(df_in["session_id_hash"])
        .shift(1)
    )

    df_in["product_count_before"] = (
        df_in.groupby(["session_id_hash", "product_raw"]).cumcount()
    )

    df_in["category_count_before"] = (
        df_in.groupby(["session_id_hash", "category_raw"]).cumcount()
    )

    df_in["is_first_product_current"] = (
        df_in["has_product"] & (df_in["product_count_before"] == 0)
    ).astype("int8")

    df_in["is_first_category_current"] = (
        df_in["has_category"] & (df_in["category_count_before"] == 0)
    ).astype("int8")

    df_in["distinct_products_before"] = (
        df_in.groupby("session_id_hash")["is_first_product_current"].cumsum()
        - df_in["is_first_product_current"]
    )

    df_in["distinct_categories_before"] = (
        df_in.groupby("session_id_hash")["is_first_category_current"].cumsum()
        - df_in["is_first_category_current"]
    )

    df_in["distinct_products_now"] = (
        df_in["distinct_products_before"] + df_in["is_first_product_current"]
    )

    df_in["distinct_categories_now"] = (
        df_in["distinct_categories_before"] + df_in["is_first_category_current"]
    )

    # Base B/D propositions
    df_in["B_prop_str"] = df_in["behavior_token"].astype(str)
    df_in["D_prop_str"] = df_in["dwell_token"].astype(str)

    # Product propositions
    prod_parts: List[np.ndarray] = []
    prod_parts.append(np.where(df_in["has_product"], "hp", "np"))

    has_prev_product = df_in["prev_product_bearing"].notna()
    same_product = has_prev_product & (df_in["product_raw"] == df_in["prev_product_bearing"])
    changed_product = has_prev_product & (df_in["product_raw"] != df_in["prev_product_bearing"])

    prod_parts.append(np.where(same_product, "sp", ""))
    prod_parts.append(np.where(changed_product, "pc", ""))

    prod_parts.append(np.where(df_in["has_product"] & (df_in["product_count_before"] == 0), "fp", ""))
    prod_parts.append(np.where(df_in["has_product"] & (df_in["product_count_before"] > 0), "rp", ""))

    product_seen_now = df_in["product_count_before"] + 1

    prod_parts.append(np.where(df_in["has_product"] & (product_seen_now >= 2), "p2", ""))
    prod_parts.append(np.where(df_in["has_product"] & (product_seen_now >= 3), "p3", ""))

    prod_parts.append(np.where(df_in["has_product"] & (df_in["distinct_products_now"] <= 1), "dp1", ""))
    prod_parts.append(np.where(df_in["has_product"] & (df_in["distinct_products_now"] == 2), "dp2", ""))
    prod_parts.append(np.where(df_in["has_product"] & (df_in["distinct_products_now"] >= 3), "dp3", ""))

    prod_mat = np.vstack(prod_parts).T

    def join_nonempty(row: Sequence[str]) -> str:
        return "+".join([x for x in row if x])

    df_in["P_prop_str"] = [join_nonempty(row) for row in prod_mat]

    # Category propositions
    cat_parts: List[np.ndarray] = []
    cat_parts.append(np.where(df_in["has_category"], "hc", ""))

    has_prev_category = df_in["prev_category_bearing"].notna()
    same_category = has_prev_category & (df_in["category_raw"] == df_in["prev_category_bearing"])
    changed_category = has_prev_category & (df_in["category_raw"] != df_in["prev_category_bearing"])

    cat_parts.append(np.where(same_category, "sc", ""))
    cat_parts.append(np.where(changed_category, "cc", ""))
    cat_parts.append(np.where(df_in["has_category"] & (df_in["category_count_before"] > 0), "rc", ""))

    cat_mat = np.vstack(cat_parts).T
    df_in["C_prop_str"] = [join_nonempty(row) for row in cat_mat]

    # Pack-level proposition sets
    df_in["comp_B"] = df_in["B_prop_str"]
    df_in["comp_BD"] = combine_prop_strings(df_in["B_prop_str"], df_in["D_prop_str"])
    df_in["comp_BP"] = combine_prop_strings(df_in["B_prop_str"], df_in["P_prop_str"], df_in["C_prop_str"])
    df_in["comp_BDP"] = combine_prop_strings(df_in["B_prop_str"], df_in["D_prop_str"], df_in["P_prop_str"], df_in["C_prop_str"])

    for col in ["comp_B", "comp_BD", "comp_BP", "comp_BDP"]:
        log(f"\n{col}: unique={df_in[col].nunique()}")
        log(str(df_in[col].value_counts().head(15)))

    return df_in, category_source


# -----------------------------
# LTLf-safe symbol sequences
# -----------------------------

def add_ltlf_symbols(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    log("[7/12] Mapping proposition sets to TLTLf-safe composite symbols")

    df_in = df_in.copy()
    symbol_maps: Dict[str, Dict[str, str]] = {}

    for pack in ["B", "BD", "BP", "BDP"]:
        col = f"comp_{pack}"
        symbol_maps[pack] = build_symbol_map(df_in[col])
        df_in[f"sym_{pack}"] = df_in[col].astype(str).map(symbol_maps[pack]).astype(str)

        log(f"\nPack {pack}: num composite symbols={len(symbol_maps[pack])}")
        for k, v in list(symbol_maps[pack].items())[:10]:
            log(f"  {v}: {k}")

    return df_in, symbol_maps


# -----------------------------
# Final sessions and splits
# -----------------------------

def build_final_sessions(
    df_in: pd.DataFrame,
    session_label: pd.DataFrame,
    prep: Path,
) -> pd.DataFrame:
    log("[8/12] Building final session-level CSV")

    session_final = (
        df_in.groupby("session_id_hash", sort=False)
        .agg(
            user_id=("user_id", "first"),
            user_session=("user_session", "first"),
            start_time=("event_time", "min"),
            end_time=("event_time", "max"),
            n_events=("event_type", "size"),

            behavior_sequence=("behavior_token", join_tokens),
            behavior_id_sequence=("behavior_id", join_tokens),
            dwell_token_sequence=("dwell_token", join_tokens),
            dwell_sec_sequence=("dwell_sec", join_float_tokens),
            product_sequence=("product_raw", join_tokens),
            category_sequence=("category_raw", join_tokens),
            brand_sequence=("brand_raw", join_tokens),

            prop_B=("comp_B", join_tokens),
            prop_BD=("comp_BD", join_tokens),
            prop_BP=("comp_BP", join_tokens),
            prop_BDP=("comp_BDP", join_tokens),

            ltlf_B=("sym_B", join_tokens),
            ltlf_BD=("sym_BD", join_tokens),
            ltlf_BP=("sym_BP", join_tokens),
            ltlf_BDP=("sym_BDP", join_tokens),
        )
        .reset_index()
    )

    session_final = session_final.merge(
        session_label[
            [
                "session_id_hash",
                "purchase",
                "full_n_events",
                "full_start_time",
                "full_end_time",
            ]
        ],
        on="session_id_hash",
        how="inner",
    )

    session_final["duration_sec"] = (
        session_final["end_time"] - session_final["start_time"]
    ).dt.total_seconds().fillna(0)

    log(f"Final sessions shape: {session_final.shape}")
    log(f"Purchase ratio: {session_final['purchase'].mean():.6f}")
    log("Length summary:")
    log(str(session_final["n_events"].describe(percentiles=[.5, .75, .9, .95, .99])))

    out = prep / "cosmetics_dec_sessions_ltl_ready.csv"
    session_final.to_csv(out, index=False)
    log(f"Saved: {out}")

    return session_final


def save_vocab_info(
    prep: Path,
    dataset_name: str,
    target_file: str,
    dwell_stats: Dict[str, object],
    category_source: str,
    symbol_maps: Dict[str, Dict[str, str]],
) -> None:
    log("[9/12] Saving vocab and symbol mapping JSON")

    vocab_info = {
        "dataset": dataset_name,
        "target_file": target_file,
        "vocab": list(CODEBOOK.keys()),
        "codebook": CODEBOOK,
        "dwell_stats": dwell_stats,
        "category_source": category_source,
        "notes": [
            "Cosmetics view is mapped to d = product detail.",
            "Purchase event is used only for session label and removed from input traces.",
            "Product_id is kept for raw neural branch but is not used directly as an LTLf proposition.",
            "Product/category symbolic evidence follows the previous dataset codebook.",
            "Dwell buckets are fitted from seed 42 train split only.",
            "Formula learning uses last 10 input events.",
            "Raw/fusion training uses first 10 input events.",
            "Model checking uses full input trace.",
        ],
        "symbol_maps": symbol_maps,
    }

    out = prep / "cosmetics_dec_vocab_and_symbol_maps.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, indent=2)

    log(f"Saved: {out}")


def merge_final_columns_to_splits(
    session_final: pd.DataFrame,
    split_dir: Path,
    seeds: Sequence[int],
) -> None:
    log("[10/12] Merging final sequence columns into split files")

    final_cols = [
        "session_id_hash",
        "behavior_sequence",
        "behavior_id_sequence",
        "dwell_token_sequence",
        "dwell_sec_sequence",
        "product_sequence",
        "category_sequence",
        "brand_sequence",
        "prop_B",
        "prop_BD",
        "prop_BP",
        "prop_BDP",
        "ltlf_B",
        "ltlf_BD",
        "ltlf_BP",
        "ltlf_BDP",
        "n_events",
        "duration_sec",
    ]

    final_lookup = session_final[final_cols].copy()

    for seed in seeds:
        seed_dir = split_dir / f"seed_{seed}"

        for split_name in ["train", "val", "test"]:
            path = seed_dir / f"{split_name}.csv"
            split = pd.read_csv(path, dtype={"session_id_hash": str, "user_id": str})

            drop_cols = [c for c in final_cols if c in split.columns and c != "session_id_hash"]
            split = split.drop(columns=drop_cols, errors="ignore")

            split = split.merge(final_lookup, on="session_id_hash", how="left")

            required_cols = [
                "behavior_sequence",
                "dwell_token_sequence",
                "product_sequence",
                "category_sequence",
                "ltlf_B",
                "ltlf_BD",
                "ltlf_BP",
                "ltlf_BDP",
            ]

            for col in required_cols:
                if split[col].isna().any():
                    missing = int(split[col].isna().sum())
                    raise AssertionError(f"Missing {col} in seed {seed} {split_name}: {missing}")

            split.to_csv(path, index=False)

            log(
                f"seed={seed} split={split_name} shape={split.shape} "
                f"pos_rate={split['purchase'].mean():.4f}"
            )


# -----------------------------
# Sanity checks and summary
# -----------------------------

def sanity_checks(session_final: pd.DataFrame, split_dir: Path, seeds: Sequence[int], prep: Path) -> None:
    log("[11/12] Running sanity checks")

    # Behavior sequence should not contain purchase.
    bad_purchase_like = session_final["behavior_sequence"].str.contains("purchase|PURCHASE", regex=True, na=False)
    if bad_purchase_like.any():
        raise AssertionError("Found purchase-like token in behavior_sequence")

    allowed_behavior_tokens = {"d", "a", "r"}
    for i, seq in enumerate(session_final["behavior_sequence"].head(1000)):
        toks = str(seq).split()
        if any(t not in allowed_behavior_tokens for t in toks):
            raise AssertionError(f"Unexpected behavior token in row {i}: {seq}")

    # Print first-vs-last examples.
    sample = session_final[session_final["n_events"] >= 15]
    if len(sample) > 0:
        sample = sample.sample(min(5, len(sample)), random_state=42)
        log("\nFirst-window vs last-window examples:")
        for _, row in sample.iterrows():
            seq = str(row["behavior_sequence"]).split()
            log(f"session={row['session_id_hash']} label={row['purchase']}")
            log(f"  first10={seq[:10]}")
            log(f"  last10 ={seq[-10:]}")

    # Leakage checks.
    for seed in seeds:
        seed_dir = split_dir / f"seed_{seed}"

        train = pd.read_csv(seed_dir / "train.csv", dtype={"session_id_hash": str, "user_id": str})
        val = pd.read_csv(seed_dir / "val.csv", dtype={"session_id_hash": str, "user_id": str})
        test = pd.read_csv(seed_dir / "test.csv", dtype={"session_id_hash": str, "user_id": str})

        assert set(train["user_id"]).isdisjoint(set(val["user_id"]))
        assert set(train["user_id"]).isdisjoint(set(test["user_id"]))
        assert set(val["user_id"]).isdisjoint(set(test["user_id"]))

        assert set(train["session_id_hash"]).isdisjoint(set(val["session_id_hash"]))
        assert set(train["session_id_hash"]).isdisjoint(set(test["session_id_hash"]))
        assert set(val["session_id_hash"]).isdisjoint(set(test["session_id_hash"]))

        log(f"Seed {seed}: leakage check passed.")

    # Split summary.
    summary_rows: List[Dict[str, object]] = []

    for seed in seeds:
        seed_dir = split_dir / f"seed_{seed}"
        for split_name in ["train", "val", "test"]:
            split = pd.read_csv(seed_dir / f"{split_name}.csv", dtype={"session_id_hash": str})
            summary_rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "sessions": len(split),
                    "positives": int(split["purchase"].sum()),
                    "negatives": int((split["purchase"] == 0).sum()),
                    "pos_rate": float(split["purchase"].mean()),
                    "users": int(split["user_id"].nunique()),
                    "mean_len": float(split["n_events"].mean()),
                    "median_len": float(split["n_events"].median()),
                    "p95_len": float(split["n_events"].quantile(0.95)),
                }
            )

    split_summary = pd.DataFrame(summary_rows)
    out = prep / "cosmetics_dec_split_summary.csv"
    split_summary.to_csv(out, index=False)

    log("\nSplit summary:")
    log(str(split_summary))
    log(f"Saved: {out}")


def print_next_steps(base: Path, prep: Path, split_dir: Path) -> None:
    log("[12/12] Done")
    log("")
    log("Generated files:")
    log(f"  sessions_csv: {prep / 'cosmetics_dec_sessions_ltl_ready.csv'}")
    log(f"  split_dir   : {split_dir}")
    log(f"  vocab_json  : {prep / 'cosmetics_dec_vocab_and_symbol_maps.json'}")
    log(f"  dwell_json  : {prep / 'cosmetics_dec_dwell_stats.json'}")
    log(f"  summary_csv : {prep / 'cosmetics_dec_split_summary.csv'}")
    log("")
    log("Important protocol reminder:")
    log("  Formula learning window : last 10 input events")
    log("  Raw/fusion training     : first 10 input events")
    log("  Model checking          : full input trace")
    log("")
    log("Expected LTLf columns:")
    log("  ltlf_B, ltlf_BD, ltlf_BP, ltlf_BDP")
    log("")
    log("Expected raw columns:")
    log("  behavior_sequence")
    log("  dwell_token_sequence")
    log("  product_sequence")
    log("  category_sequence")


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Cosmetics Shop dataset for LTLf clickstream experiments."
    )

    parser.add_argument(
        "--base",
        type=str,
        default="/content/drive/MyDrive/ltlf_cosmetics_experiments",
        help="Base output directory.",
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to input CSV. If omitted and --download_kaggle is set, downloads from Kaggle.",
    )

    parser.add_argument(
        "--download_kaggle",
        action="store_true",
        help="Download Kaggle dataset via kagglehub and use --target_file.",
    )

    parser.add_argument(
        "--target_file",
        type=str,
        default="2019-Dec.csv",
        help="Target file name inside Kaggle dataset.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cosmetics_dec",
        help="Dataset name used in metadata.",
    )

    parser.add_argument(
        "--neg_ratio",
        type=int,
        default=4,
        help="Random negative sampling ratio per split.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Split seeds.",
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional debug mode: read only first N rows.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(args.base)
    prep = base / f"{args.dataset_name}_ready"
    split_dir = prep / f"{args.dataset_name}_splits"

    base.mkdir(parents=True, exist_ok=True)
    prep.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    log("============================================================")
    log("Prepare Cosmetics Dataset for LTLf Clickstream Experiments")
    log("============================================================")
    log(f"base        : {base}")
    log(f"prep        : {prep}")
    log(f"split_dir   : {split_dir}")
    log(f"dataset_name: {args.dataset_name}")
    log(f"target_file : {args.target_file}")
    log(f"seeds       : {args.seeds}")
    log(f"neg_ratio   : {args.neg_ratio}")
    log("")

    if args.csv_path:
        csv_path = Path(args.csv_path)
    elif args.download_kaggle:
        csv_path = download_kaggle_dataset(args.target_file)
    else:
        raise ValueError("Provide --csv_path or use --download_kaggle")

    df = load_cosmetics_csv(csv_path, nrows=args.nrows)

    session_label, df_in = build_labels_and_input(df)

    # Free original df memory early.
    del df
    gc.collect()

    session_base = build_preliminary_sessions(df_in, session_label, prep)
    build_splits(session_base, split_dir, args.seeds, args.neg_ratio)

    df_in, dwell_stats = add_dwell_tokens(df_in, split_dir, prep)
    df_in, category_source = add_product_category_props(df_in)
    df_in, symbol_maps = add_ltlf_symbols(df_in)

    session_final = build_final_sessions(df_in, session_label, prep)

    save_vocab_info(
        prep=prep,
        dataset_name=args.dataset_name,
        target_file=args.target_file,
        dwell_stats=dwell_stats,
        category_source=category_source,
        symbol_maps=symbol_maps,
    )

    merge_final_columns_to_splits(session_final, split_dir, args.seeds)
    sanity_checks(session_final, split_dir, args.seeds, prep)
    print_next_steps(base, prep, split_dir)


if __name__ == "__main__":
    main()
