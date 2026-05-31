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

`ui/index.html` reads a bundled `ui/data.js` (written by the driver) — open it
directly, no server. It shows the outcome matrix, a citation ledger across the
whole run, a per-question table, and a detail panel that flags each invented or
misattributed citation. *(Screenshot uses the scripted fixture; real numbers
land once a model is wired.)*

## Status

Built **platform-agnostic first**: the core (`env.py`) runs and is testable
without mesocosm. The mesocosm adapter is a thin shim added once we have the
real scaffold. See issues for the task breakdown (milestones M1 core / M2
mesocosm / M3 optional).

## Data

[HotpotQA](https://hf.co/datasets/hotpotqa/hotpot_qa) distractor setting: each
question ships with gold supporting paragraphs + distractor paragraphs. We
treat one paragraph as one citable passage.

> Note: HotpotQA is natively 2-hop. We stratify by question `type`
> (bridge/comparison), not by hop count.

- `data/corpus.json` — `{passage_id: text}`
- `data/questions.json` — `[{qid, question, gold_answer, gold_support_ids, hop, type, level}]`

## Stack

Python 3.9+ · `rank_bm25` (keyword retrieval) · DeepSeek API (system under test,
intentionally cheap/error-prone) · vanilla JS trace UI. LLM-judge faithfulness
(Gemini Flash) is an optional upgrade, off by default.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install rank_bm25
python scripts/build_data.py        # regenerate data/ from HotpotQA
python -m pytest                     # scoring + env tests
```

## Layout

```
env.py              # core: data load + BM25 search + scoring + reset/step/close
adapter.py          # mesocosm bridge (added with the real scaffold)
benchanything.json  # mesocosm env manifest
data/               # corpus.json + questions.json
scripts/build_data.py
agent/              # DeepSeek agent + local driver
runs/               # traces + summary.json
ui/index.html       # trace visualizer
```
