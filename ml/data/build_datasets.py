#!/usr/bin/env python3
"""Build the P0 training datasets from seed-labelled exports (scripts/build_training_set.py output).

Makes the September hand-applied filtering reproducible:
  * keeps only the three in-scope classes the transformer was trained on
  * drops Pot A rows before POT_A_CUTOFF (corroded soil-moisture sensor; see
    docs/plan/nn-inference-upgrade-plan.md, Phase 1b)
  * sorts chronologically and records the 80/20 chronological split point, identical to
    scripts/train_transformer.py, so every framework trains and tests on the same rows

Datasets:
  sept15     data-from-board/sept15 (2026-08-14 -> 2026-09-15). Reproduces the shipped model's data.
  combined   sept15 + Oct05 (raw 1 Hz readings 2026-09-18 -> 2026-10-05). Larger v2 run.
             The gap 2026-09-15 -> 09-18 has no raw readings in either export.

Usage:
  .venv/bin/python ml/data/build_datasets.py \
      --sept15 data-local/datasets/sept15_raw.jsonl \
      --oct05  data-local/datasets/oct05_raw.jsonl \
      --out-dir data-local/datasets
"""

import argparse
import json
import os

POT_A_CUTOFF = "2026-09-10T08:23:00"   # UTC. Pot A rows before this are excluded (corroded sensor).
IN_SCOPE = ("stable", "waterlogged_risk", "soil_health_degrading")
TRAIN_FRACTION = 0.8


def load_filtered(path: str) -> tuple[list[dict], dict]:
    rows, stats = [], {"read": 0, "out_of_scope": 0, "pot_a_before_cutoff": 0}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["read"] += 1
            if r["label"] not in IN_SCOPE:
                stats["out_of_scope"] += 1
                continue
            if r["pot"] == "A" and r["timestamp"][:19] < POT_A_CUTOFF:
                stats["pot_a_before_cutoff"] += 1
                continue
            rows.append(r)
    return rows, stats


def write_dataset(name: str, rows: list[dict], sources: dict, out_dir: str) -> dict:
    rows.sort(key=lambda r: r["timestamp"])
    split = int(len(rows) * TRAIN_FRACTION)
    data_path = os.path.join(out_dir, f"{name}.jsonl")
    with open(data_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts: dict[str, int] = {}
    for r in rows:
        k = f"{r['pot']}:{r['label']}"
        counts[k] = counts.get(k, 0) + 1
    meta = {
        "name": name,
        "rows": len(rows),
        "split_index": split,
        "train_rows": split,
        "test_rows": len(rows) - split,
        "split_timestamp": rows[split]["timestamp"] if rows else None,
        "first_timestamp": rows[0]["timestamp"] if rows else None,
        "last_timestamp": rows[-1]["timestamp"] if rows else None,
        "pot_a_cutoff_utc": POT_A_CUTOFF,
        "in_scope_labels": list(IN_SCOPE),
        "counts_by_pot_label": dict(sorted(counts.items())),
        "sources": sources,
    }
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sept15", required=True, help="seed-labelled jsonl built from the sept15 export")
    ap.add_argument("--oct05", help="seed-labelled jsonl built from the Oct05 export (enables 'combined')")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sept_rows, sept_stats = load_filtered(args.sept15)
    meta = write_dataset("sept15", list(sept_rows), {"sept15": sept_stats}, args.out_dir)
    print(json.dumps(meta, indent=2))

    if args.oct05:
        oct_rows, oct_stats = load_filtered(args.oct05)
        overlap_from = min(r["timestamp"] for r in oct_rows) if oct_rows else None
        sept_last = max(r["timestamp"] for r in sept_rows) if sept_rows else None
        if overlap_from and sept_last and overlap_from <= sept_last:
            raise SystemExit(f"Oct05 rows start {overlap_from} before sept15 ends {sept_last}; refusing to double-count")
        meta = write_dataset("combined", sept_rows + oct_rows,
                             {"sept15": sept_stats, "oct05": oct_stats}, args.out_dir)
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
