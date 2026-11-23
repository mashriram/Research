"""Regenerates the human-readable layer of benchmark_results/ from benchmarks/catalog.py:

  benchmark_results/README.md                    index of every benchmark (headline, status, paper location)
  benchmark_results/<group>/<name>/README.md     per-benchmark page: what / why / how to run / files / results table
  benchmarks/README.md                           map of every script -> benchmark -> results folder

Run from the hsnlm/ root:  python benchmarks/build_results_index.py [--root DIR]
Idempotent: only README files are written; result files are never modified.
"""
import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalog import ARCHIVE, CATALOG, PAPER_MODEL, SUPPORT, folder  # noqa: E402

PREFERRED = ["val_bpb_full", "val_bpb", "val_bpb_len2048", "val_bpb_len4096", "val_bpb_len8192", "val_bpb_len16384",
             "test_acc", "acc_T64", "acc_T128", "acc_T256", "acc_T512", "steps_to_solve", "reached_len",
             "ms_per_step", "ms", "tok_s", "max_batch", "cache_mb", "peak_mem_mb", "train_s", "lr", "params"]
MAX_COLS, MAX_ROWS = 7, 120


def docstring(path):
    try:
        return (ast.get_docstring(ast.parse(open(path).read())) or "").strip()
    except (OSError, SyntaxError):
        return ""


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return "x" if v is None else str(v)


def results_table(path):
    try:
        data = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        return f"_Could not parse ({type(e).__name__})._"
    if not isinstance(data, dict):
        return "_Non-tabular JSON; open the file directly._"
    rows = {k: v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")}
    if not rows:
        return "_No per-run records; open the file directly._"
    count = {c: sum(1 for r in rows.values() if isinstance(r.get(c), (int, float)) or (c == "steps_to_solve" and c in r)) for c in PREFERRED}
    cols = [c for c in PREFERRED if count[c] > 0][:MAX_COLS]
    if not cols:
        scalar = {}
        for r in rows.values():
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    scalar[k] = scalar.get(k, 0) + 1
        cols = sorted(scalar, key=lambda k: -scalar[k])[:MAX_COLS]
    if not cols:
        return "_Nested records; open the file directly._"
    lines = ["| run | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for i, (k, r) in enumerate(rows.items()):
        if i == MAX_ROWS:
            lines.append(f"| ... {len(rows) - MAX_ROWS} more rows in the JSON | " + " | " * (len(cols) - 1) + " |")
            break
        lines.append(f"| `{k}` | " + " | ".join(fmt(r.get(c, "")) if c in r else "" for c in cols) + " |")
    return "\n".join(lines)


def bench_readme(e, root):
    d = os.path.join(root, folder(e))
    os.makedirs(d, exist_ok=True)
    out = [f"# {e['title']}", "",
           f"**Status:** {e['status']}  ·  **Paper:** {e.get('paper', 'not in paper (development history)')}  ·  "
           f"**Script(s):** " + ", ".join(f"[`{s}`](../../../benchmarks/{s})" for s in e["scripts"]), ""]
    if e.get("notes"):
        out += ["## Summary", "", e["notes"], ""]
    doc = docstring(os.path.join(root, "benchmarks", e["scripts"][0]))
    if doc:
        out += ["## What it tests and how (from the script)", "", "```text", doc, "```", ""]
    out += ["## Reproduce", "", "From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):", "",
            "```bash", *e["run"], "```", ""]
    files = [f for f in e["files"] if os.path.exists(os.path.join(d, f))]
    missing = [f for f in e["files"] if f not in files]
    if files or missing:
        out += ["## Files", "", "| file | kind | size |", "|---|---|---|"]
        for f in files:
            kind = {"json": "results (machine-readable)", "log": "stdout of the run", "md": "write-up"}.get(f.rsplit(".", 1)[-1], "")
            out.append(f"| [`{f}`]({f}) | {kind} | {os.path.getsize(os.path.join(d, f)) / 1024:,.1f} KB |")
        for f in missing:
            out.append(f"| `{f}` | not present | |")
        out.append("")
    arch = [f for f, n in ARCHIVE.items() if n == e["name"]]
    if arch:
        out += ["Archived (invalid, do not use): " + ", ".join(f"[`{f}`](../../_archive/{f})" for f in arch), ""]
    for f in files:
        if f.endswith(".json"):
            out += [f"## Results: `{f}`", "", "Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; "
                    "`x` = not solved). Paper tables aggregate these over seeds.", "", results_table(os.path.join(d, f)), ""]
    open(os.path.join(d, "README.md"), "w").write("\n".join(out))


def index(root):
    out = ["# Benchmark results", "",
           f"Every benchmark behind the DEEPA paper, one folder each, with its raw logs, machine-readable JSON and a README "
           f"(what it tests, how to reproduce, results). Paper model: {PAPER_MODEL}.", "",
           "* Aggregated paper tables: [PHASE_B_RESULTS.md](PHASE_B_RESULTS.md) (reviewer response, all new tables) and "
           "[PHASE_A_RESULTS.md](PHASE_A_RESULTS.md) (first round).",
           "* Scripts and how they map to these folders: [../benchmarks/README.md](../benchmarks/README.md).",
           "* `_archive/`: result files from runs with known bugs; never use them. `_queue_logs/`: orchestration logs of the GPU queues.", ""]
    for group, head in (("paper", "Paper benchmarks (numbers used in the manuscript)"), ("development", "Development history (how the paper model was found)")):
        out += [f"## {head}", "", "| benchmark | paper location | status |", "|---|---|---|"]
        for e in (x for x in CATALOG if x["group"] == group):
            out.append(f"| [{e['title']}]({group}/{e['name']}/README.md) | {e.get('paper', '-')} | {e['status']} |")
        out.append("")
    open(os.path.join(root, "benchmark_results", "README.md"), "w").write("\n".join(out))


def scripts_map(root):
    owner = {}
    for e in CATALOG:
        for s in e["scripts"]:
            owner.setdefault(s, []).append(e)
    out = ["# Benchmarks", "",
           "All scripts run from the `hsnlm/` root (`python benchmarks/<script>.py`). Results go to "
           "`benchmark_results/<group>/<benchmark>/`; each folder has a README with the exact command and the results.", "",
           "## Benchmark scripts", "", "| script | benchmark(s) | results folder |", "|---|---|---|"]
    for s in sorted(owner):
        out.append(f"| [`{s}`]({s}) | " + "; ".join(e["title"] for e in owner[s]) + " | " +
                   ", ".join(f"[{e['name']}](../{folder(e)}/README.md)" for e in owner[s]) + " |")
    out += ["", "## Support scripts", "", "| script | purpose |", "|---|---|"]
    out += [f"| [`{s}`]({s}) | {d} |" for s, d in SUPPORT.items()]
    out += ["", "## GPU queues", "",
            "`queues/` holds the shell queues that ran these benchmarks back to back on the single 8 GB GPU (kept as an exact record "
            "of run order). Their logs are in `benchmark_results/_queue_logs/`.", "",
            "## Reproducing the paper end to end", "",
            "```bash",
            "uv venv && uv pip install -e .        # torch 2.6 + cu124, flash-linear-attention 0.6, triton 3.2 (see pyproject)",
            "python benchmarks/data_prep/build_wikitext103_bytes.py && python -I benchmarks/data_prep/build_domains.py .",
            "bash benchmarks/queues/queue10_phaseA.sh   # then queue12, 13, 14, 15 (each waits for the previous one)",
            "python benchmarks/summarize_phaseB.py      # rebuild the aggregated tables",
            "python benchmarks/build_results_index.py   # rebuild these READMEs",
            "```"]
    open(os.path.join(root, "benchmarks", "README.md"), "w").write("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    for e in CATALOG:
        bench_readme(e, a.root)
    index(a.root)
    scripts_map(a.root)
    print(f"wrote {len(CATALOG)} benchmark READMEs + 2 indexes under {a.root}")
