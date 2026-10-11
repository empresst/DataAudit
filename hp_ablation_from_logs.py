#!/usr/bin/env python3
"""
hp_ablation_from_logs.py  (v2 – more robust parser)
"""

import re
from pathlib import Path
import pandas as pd

import os

LOG_DIR = Path(os.environ.get("LOG_DIR", Path(__file__).resolve().parent / "logs" / "ml4h_clean_v2"))

FILES = {
    ("Neural", "matched"):  LOG_DIR / "script17_lr3e-3.log",
    ("Neural", "unequal"):  LOG_DIR / "script17.log",
    ("Trees",  "matched"):  LOG_DIR / "trees_eta0.03.log",
    ("Trees",  "unequal"):  LOG_DIR / "script30.log",
}

NI_MARGIN = 0.015

def parse_one(path: Path, family: str, setting: str):
    text = path.read_text()
    rows = []

    label_pat = re.compile(r"^(vasopressor|intubation|mortality)\s*(\(incident\))?", re.I | re.M)
    era_pat   = re.compile(r"((?:20)?14 - 2016|(?:20)?17 - 2019|(?:20)?20 - 2022)")
    diff_pat  = re.compile(
        r"^\s*B(?:_xgb)?\s*[−\-]\s*C(?:_xgb)?\s+seed\s+([+-]?\d+\.\d+).*?"
        r"90%lo\s+([+-]?\d+\.\d+)\s*→\s*(\S+)",
        re.I | re.M
    )

    for block in re.split(r"\n(?=vasopressor|intubation|mortality)", text, flags=re.I):
        mlab = label_pat.search(block)
        if not mlab:
            continue
        current_label = mlab.group(1).lower()

        eras = list(era_pat.finditer(block))
        for i, mera in enumerate(eras):
            raw_era = mera.group(1)
            if raw_era.startswith("14"):
                era = "2014 - 2016"
            elif raw_era.startswith("17"):
                era = "2017 - 2019"
            elif raw_era.startswith("20 -"):
                era = "2020 - 2022"
            else:
                era = raw_era

            start = mera.end()
            end = eras[i + 1].start() if i + 1 < len(eras) else len(block)
            section = block[start:end]
            mdiff = diff_pat.search(section)
            if mdiff:
                mean = float(mdiff.group(1))
                lo90 = float(mdiff.group(2))
                verd = mdiff.group(3).lower().rstrip(".")
                rows.append({
                    "family": family,
                    "setting": setting,
                    "label": current_label,
                    "era": era,
                    "BmC": mean,
                    "lo90": lo90,
                    "verdict": verd,
                })
    return rows


def main():
    all_rows = []
    for (fam, setting), path in FILES.items():
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        all_rows.extend(parse_one(path, fam, setting))

    if not all_rows:
        raise SystemExit("No rows parsed")

    df = pd.DataFrame(all_rows)
    # keep only primary tasks
    df = df[df["label"].isin(["vasopressor", "intubation"])]

    # pivot
    pivot = df.pivot_table(
        index=["family", "label", "era"],
        columns="setting",
        values=["BmC", "lo90", "verdict"],
        aggfunc="first"
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot["same_NI"] = pivot["verdict_matched"] == pivot["verdict_unequal"]

    print("\n===== Hyper-parameter ablation (B−C) =====\n")
    print(pivot.to_string(index=False))
    print(f"\nCells with identical NI verdict: {pivot['same_NI'].sum()} / {len(pivot)}")
    pivot.to_csv("hp_ablation_summary.csv", index=False)
    print("Wrote hp_ablation_summary.csv")


if __name__ == "__main__":
    main()