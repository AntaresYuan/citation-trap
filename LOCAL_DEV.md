# Local development (Ollama)

Iterate on `files/env.py` and `files/benchanything.json` on your machine before `mesocosm env submit`. No API keys — only [Ollama](https://ollama.com).

## One-time setup

1. Install the CLI: `pip install swecc-mesocosm`
   This covers `mesocosm run local`, `bench_common` for `files/adapter.py`, and the HTTP stack (`fastapi`, `uvicorn`). You do **not** need `pip install -r requirements.txt` for the default scaffold — that file is only for extra packages your env imports (see comments in `requirements.txt`). The platform installs it when you `env submit`.
2. Install Ollama and pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Ensure Ollama is running (`ollama serve` — the desktop app usually does this).

## Dev loop

```bash
export MESOCOSM_LOCAL=1   # optional: bench-api :8010, adapter :8765 defaults
mesocosm doctor --local   # verify adapter (8765) before run local
```

**Option A — automatic adapter**

```bash
mesocosm run local
# reads files/benchanything.json; starts files/adapter.py if nothing listens on --env-url
```

**Option B — manual adapter**

**Terminal 1 — env server**

```bash
python files/adapter.py
# → http://localhost:8765/health
```

**Terminal 2 — bench episodes**

```bash
mesocosm run local
# same as: mesocosm run local --model ollama/llama3.2
```

Uses `files/benchanything.json` for the binding vow and scoring. Does **not** register the domain or create platform runs.

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `ollama/llama3.2` | Must be `ollama/<name>` matching a pulled model |
| `--episodes` | `5` | Number of episodes |
| `--env-url` | `http://localhost:8765` | Adapter URL if you changed the port |
| `--manifest` | `files/benchanything.json` | Alternate manifest path |
| `--system-prompt` | — | Extra instruction for the agent |

## Structured outputs and local dev

The platform uses **provider-native structured outputs** (enforced JSON Schema) for GPT, Claude, and Gemini when the action space is `discrete`, `continuous`, `json`, or `composite`. This guarantees the model returns exactly what your schema describes — no free-text parsing.

**Ollama does not support structured outputs.** When running `mesocosm run local`, the platform falls back to the free-text path: the model's reply is parsed heuristically (enum matching, JSON extraction). This means:

- Your `step()` may receive a slightly different format than what cloud runs deliver (e.g. a quoted string vs a bare value)
- Local accuracy results are only approximate — always validate with a cloud model before publishing

If you want to test the exact structured-output behavior locally, run the platform API locally (`docker compose up bench-api`) and use `mesocosm run create --model openai/gpt-4o-mini` against your submitted dev env.

## Ship to Mesocosm

When local runs look good:

```bash
mesocosm auth login
mesocosm env submit --name "My env" --github-url https://github.com/you/your-repo
# submit clones the repo and registers a draft domain from benchanything.json — no separate register step
mesocosm env list   # note domain_id when status is ready
mesocosm run create --domain DOMAIN_ID --vow-version 1.0.0 --model gemini/gemini-3.1-flash-lite ...
```

Platform runs use cloud models on SWECC infrastructure; local Ollama is only for your machine.

**Non-interactive auth:** `mesocosm auth login` prompts for credentials. In CI, set `SWECC_BENCH_TOKEN` or use `mesocosm auth guest`.

**Legacy:** repos that use `domain.py` with `DOMAIN_CONFIG` (not created by `mesocosm init`) can still run `mesocosm register path/to/domain.py [--auto-id] [--publish]` to POST the domain manually.

**Legacy layout:** older scaffolds kept `benchanything.json` at the repo root. `mesocosm run local --manifest benchanything.json` still works; `env submit` accepts either root or `files/benchanything.json`.
