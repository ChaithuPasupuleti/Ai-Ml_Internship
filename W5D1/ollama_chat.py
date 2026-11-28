#!/usr/bin/env python3
"""
Simple wrapper around Ollama's /api/chat endpoint.
- Sends a system prompt + user prompt.
- Returns the assistant's reply (max 3 sentences by default).
"""

import os
import sys
import argparse
import requests
from pathlib import Path

# ----------------------------------------------------------------------
# Configurable defaults (can be overridden via env / CLI)
# ----------------------------------------------------------------------
DEFAULT_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
SYSTEM_PROMPT = "You are a concise technical assistant. Answer in at most 3 sentences."


def ask(prompt: str, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL) -> str:
    """Send a single prompt to Ollama and return the assistant's text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        # num_gpu 0 = run on CPU (avoids the garbage-output GPU problem)
        "options": {"temperature": 0.7, "num_gpu": 0},
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "[no content]")
    except requests.exceptions.RequestException as exc:
        sys.stderr.write(f"Request failed: {exc}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run local LLM inference via Ollama."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Ollama model tag (e.g., llama3.2:3b)"
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL, help="Ollama API endpoint"
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file with one prompt per line (optional).",
    )
    args = parser.parse_args()

    prompts = [
        "What is a REST API?",
        "Explain Python decorators simply.",
        "What is the difference between SQL and NoSQL?",
        "What is overfitting in machine learning?",
        "Write a Python function to reverse a string.",
    ]

    if args.prompt_file and args.prompt_file.is_file():
        prompts = [
            line.strip()
            for line in args.prompt_file.read_text().splitlines()
            if line.strip()
        ]

    for i, p in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: {p}")
        print(ask(p, model=args.model, url=args.url))


if __name__ == "__main__":
    main()