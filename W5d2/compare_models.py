import os
import requests
from requests.adapters import HTTPAdapter, Retry

URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODELS = ["llama3.2:3b", "qwen2.5:3b"]
SYSTEM_PROMPT = (
    "You are a friendly Python tutor for beginners. "
    "Answer in simple words, in at most 4 sentences, "
    "and end with one short example."
)
QUESTIONS = [
    "What is a list in Python?",
    "What is the difference between a list and a tuple?",
    "How does a for loop work?",
]

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))


def ask(model, question):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    try:
        resp = session.post(URL, json=payload, timeout=300)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content")
        if not content:
            raise ValueError("Empty or unexpected response")
        return content.strip()
    except (requests.RequestException, ValueError) as e:
        return f"[Error: {e}]"


if __name__ == "__main__":
    lines = ["# Model comparison: llama3.2:3b vs qwen2.5:3b\n"]
    for q in QUESTIONS:
        lines.append(f"## Question: {q}\n")
        for m in MODELS:
            print(f"Asking {m}: {q}")
            answer = ask(m, q)
            print(answer, "\n" + "-" * 40)
            lines.append(f"### {m}\n\n{answer}\n")
    with open("comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved answers to comparison.md")