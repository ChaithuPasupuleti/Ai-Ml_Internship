# W5D1: Running LLMs Locally with Ollama

Setup and first run of local LLM inference using Ollama, a Python script that calls the Ollama API, and a comparison of llama3.2:3b and qwen2.5:3b.

## Setup

1. Install Ollama on Windows from ollama.com/download.
2. Pull both models:

```
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

3. Install the Python dependency:

```
pip install requests
```

## Files

| File | Purpose |
|---|---|
| ollama_chat.py | Calls the Ollama /api/chat endpoint with a custom system prompt and tests 5 prompts |
| compare_models.py | Runs the same 3 questions on both models and saves the results |
| comparison_output.txt / .json | Saved output of the comparison run |
| COMPARISON.md | My observations on accuracy, clarity, length and speed |
| cia_LOG.md | Log of my CIA (Full Stack Mentor Mode) interactions |
| output_5_prompts.png | Screenshot of the 5-prompt run |
| output_comparison.png | Screenshot of the comparison run |

## How to run

Make sure Ollama is running in the background, then:

```
python ollama_chat.py
python compare_models.py
```

Optional: `python ollama_chat.py --model qwen2.5:3b` runs the 5 prompts on another model.

## Notes

- The scripts use `"num_gpu": 0`, so the models run on CPU. With GPU enabled on my machine, the model produced garbled output, and the CPU setting fixed it.
- The system prompt limits answers to 3 sentences.
- Timing for the first call to each model includes model loading time, so the recursion row in the comparison is not a fair speed test.

## Summary of results

qwen2.5:3b gave shorter and faster answers on the list vs tuple and Git questions. llama3.2:3b gave longer answers with examples, such as the nested boxes analogy for recursion. Both were correct on all 3 questions. See COMPARISON.md for details.