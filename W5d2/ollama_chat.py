import os
import requests
from requests.adapters import HTTPAdapter, Retry

URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
SYSTEM_PROMPT = (
    "You are a friendly Python tutor for beginners. "
    "Answer in simple words, in at most 4 sentences, "
    "and end with one short example."
)

PROMPTS = [
    "What is a list in Python?",
    "Explain what a function is.",
    "What is the difference between a list and a tuple?",
    "What is a virtual environment?",
    "How does a for loop work?",
]

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))


def ask(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        resp = session.post(URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content")
        if content is None:
            raise ValueError("Unexpected response structure")
        return content.strip()
    except (requests.RequestException, ValueError) as e:
        return f"[Error] {e}"


if __name__ == "__main__":
    for i, p in enumerate(PROMPTS, 1):
        print(f"\n--- Prompt {i}: {p}\n")
        print(ask(p))
        print("-" * 40)