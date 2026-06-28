# Robin — Fine-tuned Qwen3 (API only)

Single flat Python project. No `src/` layout, no tests, no linter, no typechecker, no CI.

## Entrypoints

| File | Purpose |
|---|---|
| `app.py` | FastAPI server — POST `/chat` endpoint |

## Commands

```sh
# Serve (localhost:8000)
.venv/bin/python app.py
```

Package manager is `uv` (Python 3.12). To add deps: `uv add <pkg>`.

## Key gotchas

- **`robin-lora/` is gitignored** — LoRA adapter weights are never committed.
- **No requirements.txt** — `.venv` is uv-managed. Paths are absolute (symlinks to `~/.local/share/uv/python/…`), so the venv is not portable.

## Architecture

- **Base model**: `Qwen/Qwen3-1.7B` with LoRA adapter from `./robin-lora`
- **Inference**: single-pass prompt generation with `<think>role\ncontent</think>` format

## Uncommitted state

`app.py` has uncommitted changes. Use `git diff` before editing.
