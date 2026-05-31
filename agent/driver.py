"""Local driver: run an agent over the benchmark without mesocosm.

This is the platform-agnostic equivalent of what the mesocosm adapter will do.
Once mesocosm is available, the platform replaces this loop; the agent + env
core stay the same.

Usage:
    python -m agent.driver --agent scripted          # no key, full 2x2
    python -m agent.driver --agent deepseek          # real SUT (needs key)
    python -m agent.driver --agent scripted --qid q1 # single question
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CitationTrapEnv

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def make_agent(kind, env):
    if kind == "scripted":
        from agent.scripted_agent import ScriptedAgent
        return ScriptedAgent(env)
    if kind == "deepseek":
        from agent.deepseek_agent import DeepSeekAgent
        return DeepSeekAgent()
    raise ValueError(f"unknown agent: {kind}")


def run_episode(env, agent, qid, index):
    obs = env.reset(qid=qid)

    # Route search through env.step so every retrieval is recorded in the trace
    # (don't hand the agent env.index.search directly — that bypasses the env
    # and the trace would only ever show the final submit).
    def search(query, k=5):
        result, _, _, _ = env.step({"type": "search", "query": query, "k": k})
        return result["results"]

    submission = agent.run(obs, search, index=index)
    score, _, _, info = env.step({
        "type": "submit",
        "answer": submission.get("answer", ""),
        "citations": submission.get("citations", []),
    })
    trace = info["trace"]
    trace["agent"] = agent.name
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="scripted", choices=["scripted", "deepseek"])
    ap.add_argument("--qid", default=None, help="run a single question")
    ap.add_argument("--judge", action="store_true",
                    help="score faithfulness with the LLM entailment judge")
    args = ap.parse_args()

    judge = None
    if args.judge:
        from agent.judge import EntailmentJudge
        judge = EntailmentJudge()
        print("entailment judge: ON")
    env = CitationTrapEnv(judge=judge)
    agent = make_agent(args.agent, env)
    os.makedirs(RUNS_DIR, exist_ok=True)

    qids = [args.qid] if args.qid else [q["qid"] for q in env.questions]
    summary = {"agent": agent.name, "n": 0, "em": 0, "faithful_sum": 0.0,
               "quadrants": {}, "questions": []}
    traces = []

    for i, qid in enumerate(qids):
        trace = run_episode(env, agent, qid, i)
        s = trace["score"]
        traces.append(trace)
        with open(os.path.join(RUNS_DIR, f"{qid}.json"), "w") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        summary["n"] += 1
        summary["em"] += s["em"]
        summary["faithful_sum"] += s["faithfulness_score"]
        summary["quadrants"][s["quadrant"]] = summary["quadrants"].get(s["quadrant"], 0) + 1
        summary["questions"].append({
            "qid": qid, "em": s["em"], "f1": s["f1"],
            "faithfulness": s["faithfulness_score"], "quadrant": s["quadrant"],
        })
        print(f"{qid}: em={s['em']} faith={s['faithfulness_score']:.2f} -> {s['quadrant']}")

    n = summary["n"] or 1
    summary["accuracy"] = round(summary["em"] / n, 4)
    summary["mean_faithfulness"] = round(summary["faithful_sum"] / n, 4)
    with open(os.path.join(RUNS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # bundle for the static UI so index.html opens via file:// (no server)
    ui_dir = os.path.join(os.path.dirname(RUNS_DIR), "ui")
    os.makedirs(ui_dir, exist_ok=True)
    with open(os.path.join(ui_dir, "data.js"), "w") as f:
        f.write("window.CITATION_TRAP = " +
                json.dumps({"summary": summary, "traces": traces,
                            "corpus": env.corpus}, ensure_ascii=False) + ";\n")

    print(f"\n{summary['n']} episodes | accuracy={summary['accuracy']} "
          f"| mean_faithfulness={summary['mean_faithfulness']}")
    print("quadrants:", summary["quadrants"])


if __name__ == "__main__":
    main()
