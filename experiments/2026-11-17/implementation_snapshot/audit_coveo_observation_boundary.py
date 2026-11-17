"""Compare protected seed-42 input sessions to a local Coveo event log."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from diagnose_coveo import sha


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "protected_first_v1_seed42_short_raw"
OUT = ROOT / "coveo_seed42_comparison"
SOURCE = Path("C:/Users/ycw/Downloads/browsing_train.csv")
PREPARED = Path("C:/Users/ycw/Downloads/browsing_draw_random4_sessions_product_compact.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--zip-member")
    parser.add_argument("--report-name", default="raw_observation_audit.json")
    args = parser.parse_args()
    test_path = RUN / "test_ids.csv"
    test = pd.read_csv(test_path, dtype={"session_id": str}).set_index("session_id")
    prepared = []
    for chunk in pd.read_csv(
        PREPARED, usecols=["session_id_hash", "purchase", "n_events", "action_sequence"],
        dtype={"session_id_hash": str}, chunksize=20_000,
    ):
        prepared.append(chunk[chunk.session_id_hash.isin(test.index)])
    prepared = pd.concat(prepared, ignore_index=True).set_index("session_id_hash").reindex(test.index)
    if prepared.purchase.isna().any() or not (prepared.purchase == test.label).all():
        raise ValueError("Prepared source labels do not match the test split")
    pieces = []
    with ExitStack() as stack:
        source = args.source
        if args.zip_member:
            archive = stack.enter_context(ZipFile(args.source))
            source = stack.enter_context(archive.open(args.zip_member))
        for chunk in pd.read_csv(
            source, usecols=["session_id_hash", "event_type", "product_action", "server_timestamp_epoch_ms"],
            dtype={"session_id_hash": str}, chunksize=200_000,
        ):
            selected = chunk[chunk.session_id_hash.isin(test.index)].copy()
            selected["row_id"] = selected.index
            selected["timestamp"] = pd.to_numeric(selected.server_timestamp_epoch_ms, errors="coerce")
            selected = selected.dropna(subset=["timestamp"])
            event_type = selected.event_type.str.lower().str.replace("_", " ", regex=False)
            selected["action"] = selected.product_action.str.lower().map({
                "detail": "d", "add": "a", "remove": "r", "purchase": "p",
            }).where(event_type == "event product")
            selected.loc[event_type.isin(["pageview", "page view"]), "action"] = "b"
            pieces.append(selected[["session_id_hash", "timestamp", "row_id", "action"]])
    raw = pd.concat(pieces, ignore_index=True).sort_values(
        ["session_id_hash", "timestamp", "row_id"], kind="mergesort"
    )
    raw["position"] = raw.groupby("session_id_hash").cumcount()
    first_purchase = raw[raw.action == "p"].groupby("session_id_hash").position.min()
    boundary = raw.session_id_hash.map(first_purchase)
    visible = raw[raw.action.isin(["b", "d", "a", "r"]) & (boundary.isna() | (raw.position < boundary))]
    actions = visible.groupby("session_id_hash").action.agg("".join)
    matches = prepared.action_sequence.map(lambda value: "".join(json.loads(value))) == actions.reindex(test.index)
    matched_raw_ids = set(raw.session_id_hash)
    label_matches = (test.index.isin(first_purchase.index) == test.label.to_numpy()) & test.index.isin(matched_raw_ids)
    result = {
        "test_ids_sha256": sha(test_path), "candidate_source_sha256": sha(args.source),
        "source": str(args.source), "archive_member": args.zip_member,
        "test_sessions": len(test), "raw_candidate_sessions_found": int(test.index.isin(matched_raw_ids).sum()),
        "exact_label_matches": int(label_matches.sum()),
        "exact_pre_purchase_action_sequence_matches": int(matches.sum()),
        "exact_label_and_action_matches": int((matches & label_matches).sum()),
        "unmatched_or_different_positive_sessions": int((~(matches & label_matches) & (test.label == 1)).sum()),
        "unmatched_or_different_negative_sessions": int((~(matches & label_matches) & (test.label == 0)).sum()),
        "candidate_has_user_id": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.report_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
