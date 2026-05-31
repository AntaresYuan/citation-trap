"""Local server for Citation Trap — serve the dashboard AND run questions live.

It serves the static `ui/` folder and exposes a tiny JSON API so the browser
can run any of the benchmark questions through the real DeepSeek agent and see
the trace + gold-based scores appear in the dashboard.

    GET  /api/questions        -> [{qid, question, answer_kind, type}, ...]
    POST /api/run  {qid:"q42"} -> a full trace object {qid, question, agent,
                                  steps, score} (same shape as runs/<qid>.json)

Because the page and the API are same-origin, there is no CORS to configure.

Usage:
    export DEEPSEEK_API_KEY=...        # required for live runs
    python serve.py                   # http://localhost:8000
    python serve.py --port 8010
"""
import argparse
import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UI_DIR = os.path.join(HERE, "ui")

from env import CitationTrapEnv
from agent.driver import run_episode

# Built once, reused across requests.
ENV = CitationTrapEnv()
_AGENT = None  # lazily created so the server starts even without a key


def get_agent():
    global _AGENT
    if _AGENT is None:
        from agent.deepseek_agent import DeepSeekAgent
        _AGENT = DeepSeekAgent()
    return _AGENT


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=UI_DIR, **kw)

    def log_message(self, fmt, *args):
        # quiet the per-request noise; keep our own run logs
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/questions":
            return self._json(200, [
                {"qid": q["qid"], "question": q["question"],
                 "answer_kind": q.get("answer_kind"), "type": q.get("type")}
                for q in ENV.questions
            ])
        return super().do_GET()  # static files from ui/

    def do_POST(self):
        if self.path != "/api/run":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            qid = req.get("qid")
            if not any(q["qid"] == qid for q in ENV.questions):
                return self._json(400, {"error": f"unknown qid: {qid!r}"})
            print(f"[run] {qid} ...", flush=True)
            trace = run_episode(ENV, get_agent(), qid, 0)
            s = trace["score"]
            print(f"[run] {qid} -> {s['quadrant']} "
                  f"(em={s['em']} faith={s['faithfulness_score']:.2f})", flush=True)
            return self._json(200, trace)
        except Exception as e:  # surface the error to the browser, keep serving
            print(f"[run] ERROR: {e!r}", flush=True)
            return self._json(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Citation Trap → http://localhost:{args.port}")
    print(f"  questions loaded: {len(ENV.questions)}")
    print(f"  DEEPSEEK_API_KEY: {'set ✓' if has_key else 'MISSING — live runs will fail'}")
    print("  Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
