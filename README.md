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
that flags each invented or misattributed citation. *(Shipped `data.js` uses the
scripted fixture; live runs replace it with real DeepSeek results.)*

We have observed the headline `correct_but_fabricated` live: the agent answers
correctly, cites a real gold passage for each fact, then **invents a citation id
for its inference step** (an id absent from the corpus) — right answer,
fabricated evidence.

## Status

Built **platform-agnostic first**: the core (`env.py`) runs and is testable
without mesocosm. The mesocosm adapter is a thin shim added once we have the
real scaffold. See issues for the task breakdown (milestones M1 core / M2
mesocosm / M3 optional).

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
treat one paragraph as one citable passage. **100 questions / ~1000 passages**,
built through quality gates in `scripts/build_data.py`:

- every span answer is verified to literally appear in one of its gold passages
  (otherwise the question is unanswerable from the corpus — dropped);
- supporting titles must all be present; each question keeps ≥1 gold + ≥1
  distractor passage;
- yes/no answers capped at ≤50% so correctness isn't trivially guessable.

Composition: 50 bridge / 50 comparison · 79 span / 21 yes-no.

> Note: HotpotQA is natively 2-hop and this split is all `level="hard"`; we
> stratify by question `type` (bridge/comparison), not by hop count.

- `data/corpus.json` — `{passage_id: text}`
- `data/questions.json` — `[{qid, question, gold_answer, gold_support_ids, hop, type, answer_kind}]`

## Stack

Python 3.9+ · `rank_bm25` (keyword retrieval) · DeepSeek API (system under test,
intentionally cheap/error-prone) · vanilla JS trace UI. LLM-judge faithfulness
(Gemini Flash) is an optional upgrade, off by default.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install rank_bm25
python scripts/build_data.py        # (re)build data/ from HotpotQA (100 Q)
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
adapter.py          # mesocosm bridge (added with the real scaffold)
benchanything.json  # mesocosm env manifest
data/               # corpus.json + questions.json (100 Q)
scripts/build_data.py
agent/              # DeepSeek agent + scripted fixture + local driver
runs/               # traces + summary.json
ui/index.html       # dashboard (reads ui/data.js)
```
