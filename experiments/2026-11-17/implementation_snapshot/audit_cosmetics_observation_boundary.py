"""Replay the recorded seed-42 test input from the public December event log."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from diagnose_coveo import sha


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1] / "work"
OUT = ROOT / "cosmetics_seed42_audit"
ZIP = WORK / "cosmetics_dec_source_download.zip"
TEST = WORK / "cosmetics_seed42_test.csv"
KEYS = ["user_id", "user_session"]
ACTIONS = {"view": "d", "cart": "a", "remove_from_cart": "r"}


def main():
    test = pd.read_csv(TEST, usecols=KEYS + [
        "session_id_hash", "purchase", "n_events", "full_n_events", "behavior_sequence"
    ], dtype={key: str for key in KEYS})
    if test.duplicated(KEYS).any():
        raise ValueError("Duplicate user/session keys in saved test split")
    pieces = []
    with ZipFile(ZIP) as archive, archive.open("2019-Dec.csv") as stream:
        for chunk in pd.read_csv(stream, usecols=KEYS + ["event_time", "event_type", "price"],
                                 dtype={key: str for key in KEYS}, chunksize=250_000):
            chunk["row_id"] = chunk.index
            matched = chunk[chunk.user_session.isin(test.user_session)].merge(
                test[KEYS + ["session_id_hash"]], on=KEYS, validate="many_to_one"
            )
            matched["event_time"] = pd.to_datetime(
                matched.event_time.str.replace(" UTC", "", regex=False), utc=True, errors="coerce"
            )
            matched = matched.dropna(subset=["event_time", "event_type", *KEYS])
            pieces.append(matched[pd.to_numeric(matched.price, errors="coerce").fillna(0) >= 0])
    events = pd.concat(pieces, ignore_index=True).sort_values(
        ["session_id_hash", "event_time", "row_id"], kind="mergesort"
    )
    is_purchase = events.event_type == "purchase"
    if set(events.loc[~is_purchase, "event_type"]) - set(ACTIONS):
        raise ValueError("Unknown retained event type")
    events["position"] = events.groupby("session_id_hash").cumcount()
    first = events[is_purchase].groupby("session_id_hash").position.min()
    first_time = events[is_purchase].groupby("session_id_hash").event_time.min()
    events["after_first_purchase"] = events.position > events.session_id_hash.map(first)
    events["strictly_later_timestamp"] = events.event_time > events.session_id_hash.map(first_time)
    inputs = events.loc[~is_purchase].copy()
    inputs["action"] = inputs.event_type.map(ACTIONS)
    inputs["input_position"] = inputs.groupby("session_id_hash").cumcount()
    observed = inputs.groupby("session_id_hash").agg(
        replay_n=("action", "size"), replay_actions=("action", " ".join),
        retained_after_first_purchase=("after_first_purchase", "sum"),
        retained_strictly_after_purchase_time=("strictly_later_timestamp", "sum"),
    )
    observed["replay_full_n"] = events.groupby("session_id_hash").size()
    observed["replay_label"] = observed.index.isin(first.index).astype(int)
    observed["first10_retained_after_purchase"] = (
        inputs[inputs.input_position < 10].groupby("session_id_hash").after_first_purchase.sum()
    )
    checked = test.merge(observed, left_on="session_id_hash", right_index=True,
                         how="left", validate="one_to_one")
    if not (
        (checked.n_events == checked.replay_n)
        & (checked.full_n_events == checked.replay_full_n)
        & (checked.purchase == checked.replay_label)
        & (checked.behavior_sequence == checked.replay_actions)
    ).all():
        raise ValueError("Replayed raw event history differs from saved test input")
    result = {
        "source_url": "https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop",
        "source_zip_sha256": sha(ZIP), "test_csv_sha256": sha(TEST),
        "test_sessions_matching_labels_counts_and_action_sequences": len(checked),
        "positive_sessions": int(checked.purchase.sum()),
        "positives_with_any_retained_event_after_first_purchase": int((checked.retained_after_first_purchase > 0).sum()),
        "positives_with_retained_event_strictly_later_than_purchase_time": int(
            (checked.retained_strictly_after_purchase_time > 0).sum()
        ),
        "positives_with_post_purchase_event_in_first_10_retained_events": int(
            (checked.first10_retained_after_purchase > 0).sum()
        ),
        "positives_with_zero_events_before_first_purchase": int((first == 0).sum()),
        "total_retained_events_after_first_purchase": int(checked.retained_after_first_purchase.sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "observation_boundary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    checked[["session_id_hash", "purchase", "n_events", "retained_after_first_purchase",
             "retained_strictly_after_purchase_time", "first10_retained_after_purchase"]].to_csv(
        OUT / "test_observation_boundary.csv.gz", index=False
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
