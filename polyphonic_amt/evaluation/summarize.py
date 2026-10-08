"""Aggregate run_eval.py JSONL files into markdown tables (mean, 95 % bootstrap CI over clips)."""
import argparse
import os

import numpy as np
import pandas as pd

COND_ORDER = ["clean", "snr40", "snr30", "snr20", "snr10", "snr5", "snr0", "vac10", "vac5", "vac0", "vac-5"]


def ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return v.mean(), *np.quantile(means, [0.025, 0.975])


def fmt(values, scale=100.0, digits=1) -> str:
    m, lo, hi = ci(values)
    return f"{m * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def cond_sort(conds):
    return sorted(conds, key=lambda c: COND_ORDER.index(c) if c in COND_ORDER else 99)


def main_table(df, systems: dict[str, str], metric: str, title: str) -> str:
    conds = cond_sort(df.condition.unique())
    lines = [f"### {title}", "", "| System | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for label, sysname in systems.items():
        row = []
        for c in conds:
            d = df[(df.condition == c) & (df.system == sysname)]
            row.append(fmt(d[metric].to_numpy()) if len(d) else "-")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def ablation_table(df, prefix: str, cond: str, title: str) -> str:
    gates = ["raw", "conf50", "conf", "mask", "conf+mask", "pmask", "conf+pmask"]
    cols = ["rpa", "rpa_gated", "precision", "recall", "f1", "vfa"]
    lines = [f"### {title}", "", "| Gate | RPA | RPA (gated) | Precision | Recall | F1 | Voicing FA |", "|---|---|---|---|---|---|---|"]
    for g in gates:
        d = df[(df.condition == cond) & (df.system == f"{prefix}:{g}")]
        if len(d):
            lines.append(f"| {g} | " + " | ".join(fmt(d[c].to_numpy()) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def note_table(df, title: str) -> str:
    conds = cond_sort(df.condition.unique())
    lines = [f"### {title}", "", "| System | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for sysname in ["crepeF:conf50", "crepe:conf50", "crepe:conf", "sep:conf", "sep:conf+mask", "sepF:conf+mask"]:
        row = []
        for c in conds:
            d = df[(df.condition == c) & (df.system == sysname)]
            row.append(fmt(d["note_f1"].to_numpy()) if len(d) and "note_f1" in d else "-")
        lines.append(f"| {sysname} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="JSONL files from run_eval.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = []
    for p in args.inputs:
        df = pd.read_json(p, lines=True)
        name = os.path.basename(p).replace("res_", "").replace(".jsonl", "")
        n_clips = df.clip.nunique()
        parts.append(f"## {name}  ({n_clips} clips, {df.condition.nunique()} conditions)\n")
        parts.append(
            main_table(
                df,
                {
                    "Plain CREPE, default range (crepeF:raw)": "crepeF:raw",
                    "Hybrid, default range (sepF + conf + mask)": "sepF:conf+mask",
                    "Plain CREPE, vocal range (crepe:raw)": "crepe:raw",
                    "Hybrid, vocal range (sep + conf + mask)": "sep:conf+mask",
                },
                "rpa",
                "Raw pitch accuracy, mir_eval standard (%)",
            )
        )
        parts.append(
            main_table(
                df,
                {
                    "Plain CREPE, default range, conf 0.5": "crepeF:conf50",
                    "Hybrid, default range (conf 0.15 + mask)": "sepF:conf+mask",
                    "Plain CREPE, vocal range, conf 0.5": "crepe:conf50",
                    "Hybrid, vocal range (conf 0.15 + mask)": "sep:conf+mask",
                },
                "rpa_gated",
                "Gated RPA: frames must be voiced AND within 50 cents (%)",
            )
        )
        for cond in cond_sort(df.condition.unique()):
            if cond in ("clean", "snr10", "snr0", "vac0", "vac-5"):
                parts.append(ablation_table(df, "crepe", cond, f"Gate ablation, plain CREPE input, vocal range, {cond}"))
                parts.append(ablation_table(df, "sep", cond, f"Gate ablation, Demucs vocals input, vocal range, {cond}"))
        if "note_f1" in df.columns and df.note_f1.notna().any():
            parts.append(note_table(df, "Note-level F1 (onset +-50 ms, pitch +-50 cents) (%)"))
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
