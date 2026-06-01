# Citation Trap

A [mesocosm](https://github.com/SWECC) benchmark environment for **SWECCathon 2026**.

An LLM agent plays a research analyst: it answers multi-hop questions over a
Wikipedia-derived corpus and **must cite a passage for every claim it makes**.
We score two things and cross them into a 2×2:

- **Primary — citation faithfulness:** do the cited passages actually support
  what the agent said?
- **Secondary — answer correctness:** is the answer right? (deterministic,
  against gold answers)

|                | citation trustworthy | citation fabricated / misattributed |
|----------------|----------------------|-------------------------------------|
| **answer right** | `ideal`            | ⚠ `correct_but_fabricated` ← the headline |
| **answer wrong** | `honest_wrong`     | `worst`                             |

The headline cell — *right answer, invented sources* — is the whole point: a
model can look correct while its evidence is fabricated.

## Trace viewer

![Citation Trap dashboard](ui/preview.png)

`ui/index.html` reads a bundled `ui/data.js`. Open it directly (no server) to
browse results, or run `serve.py` to **run any question live in the browser**
(▶ Run live per question, ▶ Run all live). It shows the outcome matrix, a
citation ledger across the whole run, a per-question table, and a detail panel
that flags each invented or misattributed citation. *(Shipped `data.js` is a
real judge-scored deepseek-chat run; live runs replace it.)*

### Model leaderboard

![Model leaderboard](ui/leaderboard.png)

Same 500 questions, same corpus, same retrieval — only the model changes
(entailment judge on). `ui/compare.html` ranks them:

| model | faithfulness | accuracy | correct-but-fabricated |
|-------|-------------:|---------:|-----------------------:|
| deepseek-chat     | **93.3%** | 72% | 15 |
| deepseek-reasoner | 92.7% | **77%** | 6 |

Both models are **about equally faithful (~93%)**; the reasoner trades a little
of that for higher accuracy (77% vs 72%). Note: an earlier 100-question run
suggested the reasoner cited *less* faithfully — scaling to 500 showed that gap
was small-sample noise. That correction is itself the point: a citation
benchmark only earns trust at scale. Add a model by dropping a row into the
`MODELS` registry (`gpt-4o-mini`, `ollama-llama3.2` slots are already there).

The headline cell, when it fires, is genuine. Example (q53): asked which war the
Livesey Hall memorial commemorates, the agent answers **"World War II"**
correctly — then pads with "World War II had over 60 million casualties" and
cites a **passage id that doesn't exist** in the corpus. Right answer,
fabricated source.

### How faithfulness is scored

A citation is `fabricated` if its `passage_id` isn't in the corpus (objective),
`unsupported` if there's no id (a coverage note — e.g. a pure deduction, *not* a
trust failure), else `faithful`/`misattributed`. That last split is decided two
ways:

- **gold-set proxy (default, offline):** faithful iff the id is in the
  question's `gold_support_ids`. Fast and deterministic, but it mislabels a
  real *supporting* passage that isn't the blessed gold id, and can't tell a
  gold passage actually supports *this* claim.
- **entailment judge (`--judge`, issue #10 — done):** an LLM decides whether
  the cited passage genuinely supports the claim, regardless of gold membership.
  Removes both proxy errors. `serve.py` enables it by default; each result
  carries `scored_by`.

## Running on mesocosm

Built **platform-agnostic first**, so the [mesocosm](https://wiki.swecc.org/Sweccathon)
adapter is the thin shim we always planned. `auxiliary/` wraps the core onto the
BenchAnything four-endpoint protocol:

- `auxiliary/env.py` — `CitationTrapEnv(BaseEnv)` with `reset(seed)` / `step(action)`;
  reuses the repo-root core for BM25 + scoring. Reward = citation faithfulness;
  answer correctness and the 2×2 quadrant ride in `info`. Generic platform agents
  tend to search forever and never submit, so the env nudges then forces a submit
  (search soft-cap 3, hard-cap 6; `max_steps` 25).
- `auxiliary/adapter.py` — `serve(CitationTrapEnv)` exposes `/health /reset /step /close`.
- `auxiliary/benchanything.json` — manifest (json action space with an explicit
  `schema_ref` so the platform enforces valid actions; `scalar` reward; primary
  metric `faithfulness`). Passes `mesocosm validate`.

```bash
python3.11 -m venv .venv311 && . .venv311/bin/activate
pip install swecc-mesocosm rank_bm25
mesocosm validate auxiliary/benchanything.json          # ok: true
python auxiliary/adapter.py --port 8765                  # /health → ok
mesocosm run local                                       # Ollama agent loop
mesocosm env submit --github-url https://github.com/AntaresYuan/citation-trap
```

Verified end-to-end through the four endpoints: `/reset` (seed-selected question)
→ `/step` search (BM25, gold ranks first) → `/step` submit (reward = faithfulness,
`info.quadrant` = e.g. `correct_but_fabricated`) → `/close`.

**Submitted and live on the platform** (`mesocosm env submit --solo`): status
`ready`. Platform runs (gold-proxy scored, 10 episodes): **Claude Sonnet 4.6
(42.5%) > GPT-4o (26.7%)** — both genuinely search and cite.

Gotchas the platform surfaced:
- The `BindingVow` schema is stricter than local `mesocosm validate` — reward
  `type` must be `scalar` (not `continuous`), `techniques` must be objects.
- **All Gemini models emit an empty `{}` action and score 0** on the same env
  where Claude/GPT-4o work — the platform doesn't enforce our action `schema_ref`
  for Gemini. Filed as a platform bug; not an env issue.

Design choices worth knowing:

- **Opaque passage ids** (`p_<hash>`) — an agent must actually retrieve a
  passage to learn its id; earlier `q1_supp_1`-style ids leaked which passages
  were gold.
- **Lenient correctness** for the 2×2 axis (EM, gold-span containment, or
  F1≥0.7); raw EM/F1 are still reported. Pure EM is too strict for a free-form
  agent ("Greenwich Village" vs "Greenwich Village, New York City").

## Data

[HotpotQA](https://hf.co/datasets/hotpotqa/hotpot_qa) distractor setting: each
question ships with gold supporting paragraphs + distractor paragraphs. We
treat one paragraph as one citable passage. **500 questions / ~5000 passages**,
built through quality gates in `scripts/build_data.py`. The single shared corpus
(~5000 passages, one BM25 index) is the difficulty: gold support competes with
thousands of look-alikes. Gates:

- every span answer is verified to literally appear in one of its gold passages
  (otherwise the question is unanswerable from the corpus — dropped);
- supporting titles must all be present; each question keeps ≥1 gold + ≥1
  distractor passage;
- yes/no answers capped at ≤50% so correctness isn't trivially guessable.

Composition: 250 bridge / 250 comparison · 421 span / 79 yes-no.

> Note: HotpotQA is natively 2-hop and this split is all `level="hard"`; we
> stratify by question `type` (bridge/comparison), not by hop count.

- `data/corpus.json` — `{passage_id: text}`
- `data/questions.json` — `[{qid, question, gold_answer, gold_support_ids, hop, type, answer_kind}]`

## Stack

Python 3.9+ · `rank_bm25` (keyword retrieval) · DeepSeek API (system under test,
intentionally cheap/error-prone; also reused as the entailment judge) · vanilla
JS trace UI. The entailment judge is on by default when served, off in the
deterministic CLI default.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install rank_bm25
python scripts/build_data.py        # (re)build data/ from HotpotQA (500 Q)
python -m pytest                    # scoring + env tests

# view results without a model (deterministic fixture, all four quadrants):
python -m agent.driver --agent scripted
open ui/index.html

# run for real, in the browser:
export DEEPSEEK_API_KEY=...
python serve.py                     # http://localhost:8000
#   → pick any question, click ▶ Run live (or ▶ Run all live)

# ...or run for real from the CLI:
python -m agent.driver --agent deepseek            # all questions
python -m agent.driver --agent deepseek --qid q1   # one question
```

## Layout

```
env.py              # core: data load + BM25 search + scoring + reset/step/close
serve.py            # local server: dashboard + /api/run for in-browser live runs
auxiliary/          # mesocosm env: env.py (BaseEnv) + adapter.py + benchanything.json
data/               # corpus.json + questions.json (500 Q)
CLAUDE.md           # conventions, commands, gotchas (for AI sessions)
scripts/build_data.py
agent/              # model registry + judge + scripted fixture + local driver
runs/               # per-model traces + summary + leaderboard
ui/index.html       # dashboard (reads ui/data.js)  ·  ui/compare.html (leaderboard)
LOCAL_DEV.md        # mesocosm local-dev guide (from `mesocosm init`)
```
