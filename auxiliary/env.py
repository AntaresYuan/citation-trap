"""Citation Trap — mesocosm BaseEnv.

A thin shim over the platform-agnostic core (repo-root env.py): BM25 retrieval +
deterministic citation-faithfulness scoring. The agent answers a multi-hop
question and must cite a passage for every external fact; reward is citation
faithfulness, with answer correctness and the 2x2 quadrant carried in info.

Two actions (action_space = json):
    {"type": "search", "query": str, "k": int?}   -> {"results": [...]}, continue
    {"type": "submit", "answer": str,              -> ends the episode
     "citations": [{"claim": str, "passage_id": str}]}

Faithfulness here uses the deterministic gold-set proxy (no external API), so the
env is self-contained on the platform. The LLM entailment judge stays a local
research option in the repo root, not a platform dependency.
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

# Load the repo-root core (env.py) under a distinct name to avoid colliding
# with this module, which the adapter imports as `env`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("citation_core", os.path.join(_ROOT, "env.py"))
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)  # type: ignore[union-attr]

# Build corpus + retrieval index once; reused across episodes.
_CORPUS, _QUESTIONS = _core.load_data()
_INDEX = _core.BM25Index(_CORPUS)
# Room to search the ~5000-passage corpus several times AND still submit.
# At 10, a thorough agent (e.g. Claude on the platform) burns all steps
# searching and never gets to submit -> 0 citations.
MAX_STEPS = 25


class CitationTrapEnv(BaseEnv):
    def __init__(self) -> None:
        self._q: dict[str, Any] | None = None
        self._steps = 0

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        # deterministic_reset: seed selects the question
        idx = (seed or 0) % len(_QUESTIONS)
        self._q = _QUESTIONS[idx]
        self._steps = 0
        return {"question": self._q["question"],
                "instructions": _core.TOOL_INSTRUCTIONS}

    def parse_action(self, action: Any) -> Any:
        # Cloud runs deliver JSON; local Ollama may deliver a string — accept both.
        if isinstance(action, str):
            try:
                return json.loads(action)
            except Exception:
                return {"type": "submit", "answer": action, "citations": []}
        return action

    def step(self, action: Any) -> StepResult:
        if self._q is None:
            raise RuntimeError("Call reset() before step()")
        self._steps += 1
        a = self.parse_action(action) or {}

        if a.get("type") == "search" and self._steps < MAX_STEPS:
            results = _INDEX.search(a.get("query", ""), int(a.get("k", 5)))
            return StepResult(observation={"results": results}, reward=0.0,
                              terminated=False, truncated=False, info={})

        # submit (or budget exhausted): score and end the episode
        score = _core.score_submission(a.get("answer", ""), a.get("citations", []),
                                       self._q, _CORPUS)
        info = {
            "qid": self._q["qid"],
            "answer_correct": str(score["answer_correct"]),
            "em": str(score["em"]), "f1": str(score["f1"]),
            "faithfulness": str(score["faithfulness_score"]),
            "quadrant": score["quadrant"],
            "citation_counts": json.dumps(score["citation_counts"]),
        }
        return StepResult(observation={"result": "done", "quadrant": score["quadrant"]},
                          reward=float(score["faithfulness_score"]),
                          terminated=True, truncated=False, info=info)
