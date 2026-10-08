"""Run every evaluation stage and write the summary tables.

    python evaluation/run_all.py --data data --out results

Each stage is resumable: run_eval.py appends to <out>/res_<stage>.jsonl and skips
(clip, condition) pairs already in the file, so a crashed or disconnected run can
simply be started again.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = {
    # name: run_eval.py arguments
    "vocalset_noise": ["--experiment", "noise", "--dataset", "vocalset", "--split", "test", "--per-singer", "6"],
    "vocadito_noise": ["--experiment", "noise", "--dataset", "vocadito", "--snr", "40", "20", "10", "0"],
    "vocadito_poly": ["--experiment", "polyphonic", "--dataset", "vocadito", "--vac", "10", "5", "0", "-5"],
}


def run_stage(name: str, data: str, out_dir: str, retries: int) -> bool:
    out = os.path.join(out_dir, f"res_{name}.jsonl")
    cmd = [sys.executable, os.path.join(HERE, "run_eval.py"), *STAGES[name], "--data", data, "--out", out]
    for attempt in range(1, retries + 1):
        if subprocess.run(cmd).returncode == 0:
            return True
        print(f"[{name}] attempt {attempt} failed, resuming", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stages", nargs="*", choices=list(STAGES), default=list(STAGES))
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ok = {s: run_stage(s, args.data, args.out, args.retries) for s in args.stages}
    inputs = [os.path.join(args.out, f"res_{s}.jsonl") for s in args.stages if os.path.exists(os.path.join(args.out, f"res_{s}.jsonl"))]
    if inputs:
        subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), "--inputs", *inputs,
                        "--out", os.path.join(args.out, "summary.md")], check=True)
    failed = [s for s, v in ok.items() if not v]
    if failed:
        raise SystemExit(f"stages failed: {failed}")


if __name__ == "__main__":
    main()
