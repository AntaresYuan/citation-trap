"""ScriptedAgent -- a FIXTURE, not a real system under test.

Its only job is to exercise the full pipeline (driver -> env -> scoring ->
trace) and to populate every cell of the 2x2 so the UI (task H) can be built
and validated WITHOUT a model or API key.

Unlike a real agent, it is allowed to peek at the gold data (via the env) so it
can deterministically synthesize controlled outcomes -- including the headline
`correct_but_fabricated` cell. The real system under test is DeepSeekAgent,
which only ever sees the observation + search().
"""

# round-robin target quadrant per question index
_QUADRANTS = ["ideal", "correct_but_fabricated", "honest_wrong", "worst"]


class ScriptedAgent:
    name = "scripted-fixture"

    def __init__(self, env):
        self.env = env  # fixture cheat channel: lets us read gold

    def run(self, obs, search, index=0):
        q = self.env.current  # gold record (fixture-only access)
        target = _QUADRANTS[index % len(_QUADRANTS)]

        # one real retrieval, so traces look realistic
        results = search(obs["question"], 5)
        retrieved_ids = [r["passage_id"] for r in results]
        gold_ids = q["gold_support_ids"]
        # a real-but-wrong passage to simulate misattribution
        distractor = next((pid for pid in retrieved_ids if pid not in gold_ids), None)

        answer_correct = target in ("ideal", "correct_but_fabricated")
        trustworthy = target in ("ideal", "honest_wrong")

        answer = q["gold_answer"] if answer_correct else "unknown"

        if trustworthy:
            citations = [{"claim": f"Support {i+1}", "passage_id": pid}
                         for i, pid in enumerate(gold_ids)]
        else:
            # mix a fabricated id with a misattributed real passage
            citations = [{"claim": "Fabricated support", "passage_id": "ghost_000"}]
            if distractor:
                citations.append({"claim": "Misattributed support",
                                  "passage_id": distractor})
        return {"answer": answer, "citations": citations}
