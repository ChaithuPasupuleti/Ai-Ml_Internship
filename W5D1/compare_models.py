#!/usr/bin/env python3
"""
Run the same set of questions against two locally served Ollama models
and store a readable txt report plus a structured JSON report.
"""

import time
import json
import argparse
from pathlib import Path

from ollama_chat import ask

MODELS = ["llama3.2:3b", "qwen2.5:3b"]
QUESTIONS = [
    "Explain recursion to a beginner.",
    "What is the difference between a list and a tuple in Python?",
    "Summarize what Git branching is in 2 sentences.",
]


def run_comparison(models, questions):
    txt_lines = []
    json_records = []

    for q in questions:
        txt_lines.append(f"\n=== {q}")
        for m in models:
            start = time.perf_counter()
            try:
                answer = ask(q, model=m)
            except (Exception, SystemExit) as exc:
                answer = f"[ERROR] {exc}"
            elapsed = time.perf_counter() - start

            words = len(answer.split())
            txt_lines.append(
                f"\n[{m}] ({elapsed:.1f} sec, {words} words)\n{answer}"
            )
            print(txt_lines[-1])

            json_records.append(
                {
                    "question": q,
                    "model": m,
                    "answer": answer,
                    "time_sec": round(elapsed, 2),
                    "word_count": words,
                }
            )
    return "\n".join(txt_lines), json_records


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Ollama models on a small prompt set."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory where the output files will be saved.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    txt_report, json_report = run_comparison(MODELS, QUESTIONS)

    txt_path = args.out_dir / "comparison_output.txt"
    txt_path.write_text(txt_report, encoding="utf-8")
    print(f"\nSaved readable report to {txt_path}")

    json_path = args.out_dir / "comparison_output.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    print(f"Saved structured JSON to {json_path}")


if __name__ == "__main__":
    main()