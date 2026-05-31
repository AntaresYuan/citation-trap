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


def make_agent(kind, env, model="deepseek-chat"):
    if kind == "scripted":
        from agent.scripted_agent import ScriptedAgent
        return ScriptedAgent(env)
    if kind == "deepseek":
        from agent.deepseek_agent import make_model_agent
        return make_model_agent(model)
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
    ap.add_argument("--model", default="deepseek-chat",
                    help="model name from the registry (with --agent deepseek)")
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
    agent = make_agent(args.agent, env, args.model)
    # per-model output dir so models don't overwrite each other
    model_dir = os.path.join(RUNS_DIR, agent.name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    qids = [args.qid] if args.qid else [q["qid"] for q in env.questions]
    summary = {"agent": agent.name, "model": agent.name, "judge": bool(judge),
               "n": 0, "em": 0, "correct": 0, "faithful_sum": 0.0,
               "quadrants": {}, "questions": []}
    traces = []

    for i, qid in enumerate(qids):
        trace = run_episode(env, agent, qid, i)
        s = trace["score"]
        traces.append(trace)
        with open(os.path.join(model_dir, f"{qid}.json"), "w") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        summary["n"] += 1
        summary["em"] += s["em"]
        summary["correct"] += 1 if s["answer_correct"] else 0
        summary["faithful_sum"] += s["faithfulness_score"]
        summary["quadrants"][s["quadrant"]] = summary["quadrants"].get(s["quadrant"], 0) + 1
        summary["questions"].append({
            "qid": qid, "em": s["em"], "f1": s["f1"],
            "faithfulness": s["faithfulness_score"], "quadrant": s["quadrant"],
        })
        print(f"{qid}: em={s['em']} faith={s['faithfulness_score']:.2f} -> {s['quadrant']}")

    n = summary["n"] or 1
    summary["accuracy"] = round(summary["correct"] / n, 4)       # lenient (2x2 axis)
    summary["em_accuracy"] = round(summary["em"] / n, 4)
    summary["mean_faithfulness"] = round(summary["faithful_sum"] / n, 4)
    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # bundle the last run for the single-run dashboard (index.html / file://)
    ui_dir = os.path.join(os.path.dirname(RUNS_DIR), "ui")
    os.makedirs(ui_dir, exist_ok=True)
    with open(os.path.join(ui_dir, "data.js"), "w") as f:
        f.write("window.CITATION_TRAP = " +
                json.dumps({"summary": summary, "traces": traces,
                            "corpus": env.corpus}, ensure_ascii=False) + ";\n")

    write_leaderboard()

    print(f"\n{summary['n']} episodes | accuracy={summary['accuracy']} "
          f"| mean_faithfulness={summary['mean_faithfulness']}")
    print("quadrants:", summary["quadrants"])


def write_leaderboard():
    """Aggregate every runs/<model>/summary.json into a leaderboard."""
    rows = []
    for name in sorted(os.listdir(RUNS_DIR)):
        path = os.path.join(RUNS_DIR, name, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            s = json.load(f)
        q = s.get("quadrants", {})
        rows.append({
            "model": s.get("model", name), "judge": s.get("judge", False),
            "n": s.get("n", 0), "accuracy": s.get("accuracy", 0),
            "mean_faithfulness": s.get("mean_faithfulness", 0),
            "ideal": q.get("ideal", 0),
            "correct_but_fabricated": q.get("correct_but_fabricated", 0),
            "honest_wrong": q.get("honest_wrong", 0), "worst": q.get("worst", 0),
        })
    # rank by faithfulness (primary), then accuracy
    rows.sort(key=lambda r: (r["mean_faithfulness"], r["accuracy"]), reverse=True)
    board = {"models": rows}
    with open(os.path.join(RUNS_DIR, "leaderboard.json"), "w") as f:
        json.dump(board, f, ensure_ascii=False, indent=2)
    ui_dir = os.path.join(os.path.dirname(RUNS_DIR), "ui")
    with open(os.path.join(ui_dir, "leaderboard.js"), "w") as f:
        f.write("window.CITATION_TRAP_LEADERBOARD = " +
                json.dumps(board, ensure_ascii=False) + ";\n")
    if len(rows) > 1:
        print("\n=== leaderboard (by faithfulness) ===")
        for r in rows:
            print(f"  {r['model']:<20} faith={r['mean_faithfulness']:.2f} "
                  f"acc={r['accuracy']:.2f} cbf={r['correct_but_fabricated']}")


if __name__ == "__main__":
    main()
