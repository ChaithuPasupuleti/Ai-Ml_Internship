# W5D2: Prompt Engineering & System Prompts with Ollama

## Files
- `ollama_chat.py`: calls the Ollama API with a custom system prompt and tests 5 prompts
- `compare_models.py`: asks llama3.2:3b and qwen2.5:3b the same 3 questions
- `comparison.md`: saved answers from both models
- `output1.png`, `output2.png`: output screenshots

## System prompt used
"You are a friendly Python tutor for beginners. Answer in simple words, in at most 4 sentences, and end with one short example."

## Model comparison: llama3.2:3b vs qwen2.5:3b

| Point | llama3.2:3b | qwen2.5:3b |
|---|---|---|
| Length | Longer, more detailed | Short and concise |
| Examples | Multi-line code blocks with comments | One-line inline examples |
| Completeness | Showed both list and tuple in the tuple question | Gave only a list example for the tuple question |
| Loop question | Showed the example and its printed output | Gave a short one-line loop |
| Following the prompt | Good, but answers were longer | Followed the "short answer" rule more strictly |

## Conclusion
llama3.2:3b gives clearer, more beginner-friendly answers with fuller examples. qwen2.5:3b is faster to read but sometimes too brief. For a beginner tutor, llama3.2:3b is the better fit.

## Issue faced
Ollama gave gibberish output on the GPU. Fixed by running it in CPU mode (`OLLAMA_LLM_LIBRARY=cpu`, then `ollama serve`).