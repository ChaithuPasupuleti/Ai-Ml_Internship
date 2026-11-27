# CIA Interactions (Full Stack Mentor Mode)

1. Code review of ollama_chat.py: suggested error handling, CLI args (--model, --prompt-file), docstring and env-var config. I applied these and kept num_gpu 0 for CPU.
2. Code review of compare_models.py: suggested try/except, JSON output, --out-dir and perf_counter. I applied these, removed emoji for Windows, and also caught SystemExit because ask() calls sys.exit.