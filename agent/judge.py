"""LLM entailment judge — turns citation faithfulness from a gold-set proxy
into a real "does the cited passage support the claim?" check.

Without the judge, a citation is `faithful` iff its passage_id is in the
question's gold_support_ids. That has two failure modes: it flags a real,
genuinely-supporting passage as `misattributed` just because it isn't the
blessed gold id (false positive), and it blesses a gold passage that doesn't
actually support that specific claim (false negative). The judge replaces the
gold-membership test with an entailment test, so neither happens.

Uses DeepSeek (the same key as the agent) — no extra credential. Results are
cached per (claim, passage) so a run never asks the same question twice.
"""
import json
import os
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"

SYSTEM = (
    "You are a strict fact-checking judge. You are given a CLAIM and a PASSAGE. "
    "Decide whether the PASSAGE directly supports the CLAIM — i.e. a reader of "
    "the passage alone could verify the claim. Ignore whether the claim is true "
    "in general; judge only what THIS passage supports. Answer with a single "
    "word: YES or NO."
)


class EntailmentJudge:
    name = "deepseek-judge"

    def __init__(self, api_key=None, model="deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        self.model = model
        self._cache = {}

    def entails(self, claim, passage):
        claim = (claim or "").strip()
        passage = (passage or "").strip()
        if not claim or not passage:
            return False
        key = (claim, passage)
        if key in self._cache:
            return self._cache[key]
        verdict = self._ask(claim, passage)
        self._cache[key] = verdict
        return verdict

    def _ask(self, claim, passage):
        body = json.dumps({
            "model": self.model, "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"CLAIM: {claim}\n\nPASSAGE: {passage}\n\nDoes the passage support the claim? YES or NO."},
            ],
        }).encode()
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=60))
        text = resp["choices"][0]["message"]["content"].strip().upper()
        return text.startswith("Y")
