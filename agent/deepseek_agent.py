"""DeepSeekAgent -- the real system under test.

A small ReAct-style loop over DeepSeek's OpenAI-compatible chat API. The model
sees the question + tool instructions and emits ONE JSON action per turn:

    {"type": "search", "query": "...", "k": 5}
    {"type": "submit", "answer": "...",
     "citations": [{"claim": "...", "passage_id": "..."}]}

We execute the action, feed back the observation, and repeat until it submits
or hits the step budget. It never sees gold data.

Set DEEPSEEK_API_KEY in the environment. Cheap models are intentional: they
make citation mistakes, which is exactly what the benchmark is meant to catch.
"""
import json
import os
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
MAX_STEPS = 6

SYSTEM = (
    "You are a research analyst. Answer the question using ONLY passages you "
    "retrieve. Every claim in your answer must be backed by a real retrieved "
    "passage_id. Respond with EXACTLY ONE JSON object per turn, no prose:\n"
    '  to search: {"type":"search","query":"...","k":5}\n'
    '  to finish: {"type":"submit","answer":"...",'
    '"citations":[{"claim":"...","passage_id":"..."}]}\n'
    "Do at least one search before submitting."
)


def _extract_json(text):
    """Pull the first balanced JSON object out of a model reply."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in reply: {text[:200]!r}")


class DeepSeekAgent:
    name = "deepseek-chat"

    def __init__(self, api_key=None, model=DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        self.model = model

    def _chat(self, messages):
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": 0.7}).encode()
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=60))
        return resp["choices"][0]["message"]["content"]

    def run(self, obs, search, index=0):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {obs['question']}"},
        ]
        last = {"answer": "", "citations": []}
        for _ in range(MAX_STEPS):
            reply = self._chat(messages)
            messages.append({"role": "assistant", "content": reply})
            try:
                action = _extract_json(reply)
            except ValueError:
                messages.append({"role": "user",
                                 "content": "Invalid format. Send one JSON action."})
                continue

            if action.get("type") == "search":
                results = search(action.get("query", ""), int(action.get("k", 5)))
                obs_text = "\n".join(f'{r["passage_id"]}: {r["text"]}' for r in results)
                messages.append({"role": "user",
                                 "content": f"Results:\n{obs_text}\nNext action?"})
            elif action.get("type") == "submit":
                return {"answer": action.get("answer", ""),
                        "citations": action.get("citations", [])}
            else:
                messages.append({"role": "user",
                                 "content": "Unknown action. Use search or submit."})
        return last  # budget exhausted without a submit
