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

MAX_STEPS = 8

# Model registry — every entry speaks the OpenAI chat-completions format, so
# DeepSeek, OpenAI, and local Ollama all work through one client. Add a row to
# benchmark another model (give the matching key env var). `name` is the label
# shown in the leaderboard.
MODELS = {
    "deepseek-chat":     {"base_url": "https://api.deepseek.com/chat/completions",
                          "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY"},
    "deepseek-reasoner": {"base_url": "https://api.deepseek.com/chat/completions",
                          "model": "deepseek-reasoner", "key_env": "DEEPSEEK_API_KEY"},
    "gpt-4o-mini":       {"base_url": "https://api.openai.com/v1/chat/completions",
                          "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
    "ollama-llama3.2":   {"base_url": "http://localhost:11434/v1/chat/completions",
                          "model": "llama3.2", "key_env": None},
}

SYSTEM = (
    "You are a research analyst. Answer the question using ONLY passages you "
    "retrieve. Respond with EXACTLY ONE JSON object per turn, no prose:\n"
    '  to search: {"type":"search","query":"...","k":8}\n'
    '  to finish: {"type":"submit","answer":"...",'
    '"citations":[{"claim":"...","passage_id":"..."}]}\n'
    "Do at least one search before submitting.\n"
    "Make `answer` as short as possible — just the entity or span that answers "
    "the question (e.g. \"Chief of Protocol\"), not a sentence.\n"
    "Cite each EXTERNAL FACT you relied on (a name, date, place, etc.) with the "
    "passage_id it came from. You do NOT need to cite pure logical steps that "
    "follow from facts you already cited — e.g. deciding which of two cited "
    "dates is earlier. Only cite claims that genuinely need a source."
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


class OpenAICompatAgent:
    """ReAct agent over any OpenAI-compatible chat endpoint."""

    def __init__(self, name="deepseek-chat", base_url=None, model=None,
                 key_env="DEEPSEEK_API_KEY", temperature=1.1):
        spec = MODELS.get(name, {})
        self.name = name
        self.base_url = base_url or spec.get("base_url")
        self.model = model or spec.get("model", name)
        key_env = spec.get("key_env", key_env) if name in MODELS else key_env
        self.api_key = os.environ.get(key_env) if key_env else None
        if key_env and not self.api_key:
            raise RuntimeError(f"{key_env} not set (needed for {name})")
        self.temperature = float(os.environ.get("CT_TEMPERATURE", temperature))

    def _chat(self, messages):
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": self.temperature}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url, data=body, method="POST",
                                     headers=headers)
        resp = json.load(urllib.request.urlopen(req, timeout=120))
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


def make_model_agent(name, temperature=1.1):
    """Build an agent for a registered model name (see MODELS)."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    return OpenAICompatAgent(name=name, temperature=temperature)


class DeepSeekAgent(OpenAICompatAgent):
    """Back-compat: defaults to deepseek-chat."""
    def __init__(self, model="deepseek-chat", temperature=1.1):
        super().__init__(name=model if model in MODELS else "deepseek-chat",
                         temperature=temperature)
