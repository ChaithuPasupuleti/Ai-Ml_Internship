# Comparison: llama3.2:3b vs qwen2.5:3b

Setup: Ollama on Windows, CPU only, same system prompt (max 3 sentences), 3 questions.

## Speed and length

| Question | llama3.2:3b | qwen2.5:3b |
|---|---|---|
| Explain recursion | 30.2 sec, 59 words | 33.8 sec, 31 words |
| List vs tuple | 48.5 sec, 57 words | 14.5 sec, 29 words |
| Git branching | 22.0 sec, 57 words | 14.4 sec, 46 words |

## Observations

- Accuracy: Both models answered all 3 questions correctly. Both described Git branches as "copies", which is a simplification but fine for a 2-sentence summary.
- Clarity: For recursion, llama used a nested boxes analogy that is easier for a beginner to picture. qwen gave a correct but plain definition. For list vs tuple, qwen explained what mutable and immutable mean, while llama showed the syntax with () and [].
- Length: qwen was shorter on every question (31, 29 and 46 words vs 59, 57 and 57). llama added examples and use cases, which were useful but made the answers longer.
- Speed: qwen was faster on list vs tuple (14.5 vs 48.5 sec) and Git branching (14.4 vs 22.0 sec). The recursion times are not a fair comparison because each model's first call includes loading time.
- Preferred model: llama for recursion because of the analogy, qwen for list vs tuple and Git branching because the answers were short, clear and faster.

## Conclusion

On my CPU-only setup, qwen2.5:3b was faster and more concise, so I would use it for quick factual answers. I would use llama3.2:3b when I want a beginner-friendly explanation with examples, even though it is slower.