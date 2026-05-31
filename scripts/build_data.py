"""Build corpus.json + questions.json for the Citation Trap benchmark.

Source: HotpotQA (distractor setting). Each question ships with ~10 context
paragraphs: a few gold supporting paragraphs + distractors. We treat one
paragraph (title + its sentences) as one citable "passage".

We pull rows via the HuggingFace datasets-server REST API so we avoid the
heavy `datasets` dependency (stdlib only here).

QUALITY GATES (a row is kept only if ALL pass):
  1. answerable:   for non yes/no answers, the gold answer must literally appear
                   in at least one gold passage -- otherwise the question can't
                   be answered from the corpus and is useless as a benchmark item.
  2. intact:       every supporting-fact title is present in the context.
  3. non-trivial:  >= 1 gold passage AND >= 1 distractor passage, all non-empty.
  4. well-formed:  non-empty question + answer; no duplicate question text.
Selection also caps yes/no answers at <= 50% so the correctness axis is not
dominated by trivially-guessable comparison questions.

Stratification note: HotpotQA is natively 2-hop and this split is all
level="hard"; we balance by `type` (bridge vs comparison) for variety and
record hop=2 for every item.
"""
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"
ROWS_API = "https://datasets-server.huggingface.co/rows"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

N_QUESTIONS = 500        # target count
FETCH_POOL = 2500        # pull a big pool; quality gates drop a large fraction
MAX_YESNO_FRAC = 0.5     # cap yes/no answers at half of the final set


def fetch_rows(length):
    """Fetch `length` rows from the datasets-server (max 100 per call)."""
    rows, offset = [], 0
    while len(rows) < length:
        batch = min(100, length - len(rows))
        params = urllib.parse.urlencode({
            "dataset": DATASET, "config": CONFIG, "split": SPLIT,
            "offset": offset, "length": batch,
        })
        req = urllib.request.Request(f"{ROWS_API}?{params}",
                                     headers={"User-Agent": "citation-trap"})
        data = json.load(urllib.request.urlopen(req, timeout=30))
        got = [r["row"] for r in data["rows"]]
        if not got:
            break
        rows.extend(got)
        offset += len(got)
    return rows


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def is_yesno(row):
    return _norm(row["answer"]) in ("yes", "no")


def gold_and_distractor_texts(row):
    """Return (gold_texts, distractor_texts) as lists of paragraph strings."""
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    gold_titles = set(row["supporting_facts"]["title"])
    gold, dist = [], []
    for title, sents in zip(titles, sentences):
        text = " ".join(s.strip() for s in sents).strip()
        if not text:
            continue
        (gold if title in gold_titles else dist).append(text)
    return gold, dist


def row_quality_problem(row):
    """Return a short reason string if the row fails a gate, else None."""
    q, a = row.get("question", "").strip(), row.get("answer", "").strip()
    if not q or not a:
        return "empty question/answer"
    context_titles = set(row["context"]["title"])
    supp_titles = set(row["supporting_facts"]["title"])
    if not supp_titles or not supp_titles.issubset(context_titles):
        return "supporting title missing from context"
    gold, dist = gold_and_distractor_texts(row)
    if not gold or not dist:
        return "no gold or no distractor passage"
    if not is_yesno(row):
        blob = _norm(" ".join(gold))
        if _norm(a) not in blob:
            return "answer not found in any gold passage (unanswerable)"
    return None


def build_one(qid, row):
    """Turn one (validated) HotpotQA row into (passages, question_record)."""
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    gold_titles = set(row["supporting_facts"]["title"])

    passages, gold_support_ids = {}, []
    for title, sents in zip(titles, sentences):
        text = " ".join(s.strip() for s in sents).strip()
        if not text:
            continue
        # Opaque, unguessable id: an agent must actually RETRIEVE a passage to
        # learn its id (earlier q1_supp_1 / q1_dist_2 ids leaked gold membership).
        pid = "p_" + hashlib.sha1(f"{qid}|{title}".encode()).hexdigest()[:10]
        if title in gold_titles:
            gold_support_ids.append(pid)
        passages[pid] = f"{title}. {text}"

    record = {
        "qid": qid,
        "question": row["question"],
        "gold_answer": row["answer"],
        "gold_support_ids": gold_support_ids,
        "hop": 2,
        "type": row.get("type"),
        "level": row.get("level"),
        "answer_kind": "yes_no" if is_yesno(row) else "span",
    }
    return passages, record


def select(valid, n):
    """Round-robin by type for variety, while capping yes/no at MAX_YESNO_FRAC."""
    buckets = {}
    for r in valid:
        buckets.setdefault(r.get("type", "?"), []).append(r)
    max_yesno = int(n * MAX_YESNO_FRAC)
    selected, yesno = [], 0
    progressed = True
    while len(selected) < n and progressed:
        progressed = False
        for key in sorted(buckets):
            bucket = buckets[key]
            # find next row we're allowed to take (respecting the yes/no cap)
            while bucket:
                r = bucket.pop(0)
                if is_yesno(r) and yesno >= max_yesno:
                    continue  # skip yes/no once the cap is hit
                selected.append(r)
                if is_yesno(r):
                    yesno += 1
                progressed = True
                break
            if len(selected) >= n:
                break
    return selected


def main():
    print(f"Fetching {FETCH_POOL} rows from {DATASET} [{CONFIG}/{SPLIT}] ...")
    pool = fetch_rows(FETCH_POOL)
    print(f"  got {len(pool)} rows")

    # --- quality gates ---
    valid, seen_q, drops = [], set(), {}
    for row in pool:
        qkey = _norm(row.get("question", ""))
        if qkey in seen_q:
            drops["duplicate question"] = drops.get("duplicate question", 0) + 1
            continue
        problem = row_quality_problem(row)
        if problem:
            drops[problem] = drops.get(problem, 0) + 1
            continue
        seen_q.add(qkey)
        valid.append(row)
    print(f"Quality gates: {len(valid)} valid / {len(pool)} fetched")
    for reason, k in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"    dropped {k:4d}  {reason}")

    if len(valid) < N_QUESTIONS:
        raise SystemExit(f"Only {len(valid)} valid rows; raise FETCH_POOL.")

    chosen = select(valid, N_QUESTIONS)

    corpus, questions = {}, []
    for i, row in enumerate(chosen, start=1):
        passages, record = build_one(f"q{i}", row)
        corpus.update(passages)
        questions.append(record)

    # --- final hard validation: never ship a broken file ---
    problems = []
    for q in questions:
        if not q["gold_support_ids"]:
            problems.append(f"{q['qid']}: empty gold_support_ids")
        for gid in q["gold_support_ids"]:
            if gid not in corpus:
                problems.append(f"{q['qid']}: gold id {gid} missing from corpus")
        if q["answer_kind"] == "span":
            blob = " ".join(corpus[g] for g in q["gold_support_ids"]).lower()
            if _norm(q["gold_answer"]) not in _norm(blob):
                problems.append(f"{q['qid']}: answer not in gold passages")
    if problems:
        raise SystemExit("DATA VALIDATION FAILED:\n  " + "\n  ".join(problems))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "corpus.json"), "w") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "questions.json"), "w") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    by_type = {}
    yesno = sum(1 for q in questions if q["answer_kind"] == "yes_no")
    for q in questions:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    print(f"\nWrote {len(questions)} questions, {len(corpus)} passages.")
    print("  type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"  answer kind: span={len(questions)-yesno}, yes_no={yesno}")
    print("  every span answer verified present in its gold passages ✓")


if __name__ == "__main__":
    main()
