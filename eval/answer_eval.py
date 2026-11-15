"""Run the probes and needs through `answer.ask` and count how the answer layer behaves.

    python eval/answer_eval.py eval/probe_queries.jsonl eval/need_queries.jsonl \\
        --output library/answer_runs.jsonl --sheet library/answer_judgements.csv

What it can count without a judge:

- **probes answered** - a probe is a question the library does not cover by
  design, so an answer to one is a likely fabrication (or a surprising find).
- **needs declined** - a need is written to be answerable from the library, so a
  decline is a likely wrong decline (or a retrieval miss upstream).
- **uncited answers** - answered, but with no valid [n] citation at all.
- **off-shelf citations** - a need's `relevant` list is domain-level (every
  chapter on the right shelves), so a cited chapter outside it is a hint that the
  answer leaned on an unrelated passage. A hint only: off-shelf can still be right.

What it cannot count is whether each claim in an answer is supported by the
passages it cites. That needs a person reading both, so `--sheet` writes one row
per answered question with the answer and the cited titles, and an empty
`supported` column (y / partly / n) to fill in.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answer  # noqa: E402
import retrieval  # noqa: E402


def load(paths: list[str]) -> list[dict]:
    entries = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    entries.append(json.loads(line))
    return entries


def summarise(runs: list[dict]) -> dict:
    """Counts per kind (probe / need) from finished runs."""
    out = {}
    for kind in sorted({r["kind"] for r in runs}):
        rows = [r for r in runs if r["kind"] == kind]
        answered = [r for r in rows if r["answered"]]
        cited = [c for r in answered for c in r["cited"]]
        off_shelf = [
            c for r in answered if r["relevant"]
            for c in r["cited"] if c["id"] not in set(r["relevant"])
        ]
        out[kind] = {
            "questions": len(rows),
            "answered": len(answered),
            "declined": len(rows) - len(answered),
            "uncited": sum(r["uncited"] for r in answered),
            "citations": len(cited),
            "off_shelf_citations": len(off_shelf),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("queries", nargs="+", help="query .jsonl files (probes, needs)")
    ap.add_argument("--output", default="library/answer_runs.jsonl")
    ap.add_argument("--sheet", default="library/answer_judgements.csv")
    args = ap.parse_args()

    entries = load(args.queries)
    runs = []
    try:
        for i, entry in enumerate(entries, 1):
            result = answer.ask(entry["query"])
            runs.append({
                "kind": entry.get("kind", "need"),
                "relevant": entry.get("relevant", []),
                **result,
            })
            mark = "answered" if result["answered"] else "declined"
            print(f"[{i}/{len(entries)}] {runs[-1]['kind']:5} {mark:8} {entry['query']}")
    finally:
        retrieval.close_clients()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(json.dumps(run, ensure_ascii=False) + "\n")

    with open(args.sheet, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "question", "answer", "cited", "supported", "notes"])
        for run in runs:
            if run["answered"]:
                titles = "; ".join(f"[{c['n']}] {c['title'] or c['id']}" for c in run["cited"])
                writer.writerow([run["kind"], run["question"], run["answer"], titles, "", ""])

    print()
    for kind, s in summarise(runs).items():
        print(f"{kind}: {json.dumps(s)}")
    print(f"\nruns -> {args.output}\nsheet to mark -> {args.sheet}")


if __name__ == "__main__":
    main()
