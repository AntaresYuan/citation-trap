"""Build corpus.json + questions.json for the Citation Trap benchmark.

Source: HotpotQA (distractor setting). Each question ships with ~10 context
paragraphs: a few gold supporting paragraphs + distractors. We treat one
paragraph (title + its sentences) as one citable "passage".

We pull rows via the HuggingFace datasets-server REST API so we avoid the
heavy `datasets` dependency (stdlib only here).

Stratification note: HotpotQA is natively 2-hop. The brief asks for
2-hop/3-hop/ambiguous tiers, which HotpotQA does not provide. We instead
stratify by (type, level) -- type in {bridge, comparison}, level in
{easy, medium, hard} -- to get variety, and record hop=2 for every item.
"""
import json
import os
import urllib.parse
import urllib.request

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"
ROWS_API = "https://datasets-server.huggingface.co/rows"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

N_QUESTIONS = 18        # target count (brief: 15-20)
FETCH_POOL = 100        # how many rows to pull before stratified selection


def fetch_rows(length):
    """Fetch `length` rows from the datasets-server (max 100 per call)."""
    rows = []
    offset = 0
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


def stratified_select(rows, n):
    """Pick n rows spread across (type, level) buckets, round-robin."""
    buckets = {}
    for r in rows:
        key = (r.get("type", "?"), r.get("level", "?"))
        buckets.setdefault(key, []).append(r)
    selected, exhausted = [], False
    while len(selected) < n and not exhausted:
        exhausted = True
        for key in sorted(buckets):
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                exhausted = False
                if len(selected) >= n:
                    break
    return selected


def build_one(qid, row):
    """Turn one HotpotQA row into (passages_for_corpus, question_record)."""
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]  # list[list[str]] parallel to titles
    gold_titles = set(row["supporting_facts"]["title"])

    passages = {}          # passage_id -> text
    gold_support_ids = []
    supp_n = dist_n = 0
    for title, sents in zip(titles, sentences):
        text = " ".join(s.strip() for s in sents).strip()
        if not text:
            continue
        if title in gold_titles:
            supp_n += 1
            pid = f"{qid}_supp_{supp_n}"
            gold_support_ids.append(pid)
        else:
            dist_n += 1
            pid = f"{qid}_dist_{dist_n}"
        # prepend the title so retrieval/UI carry the source name
        passages[pid] = f"{title}. {text}"

    record = {
        "qid": qid,
        "question": row["question"],
        "gold_answer": row["answer"],
        "gold_support_ids": gold_support_ids,
        "hop": 2,
        "type": row.get("type"),
        "level": row.get("level"),
    }
    return passages, record


def main():
    print(f"Fetching {FETCH_POOL} rows from {DATASET} [{CONFIG}/{SPLIT}] ...")
    pool = fetch_rows(FETCH_POOL)
    print(f"  got {len(pool)} rows")

    chosen = stratified_select(pool, N_QUESTIONS)
    print(f"Selected {len(chosen)} questions across (type, level) buckets")

    corpus, questions = {}, []
    for i, row in enumerate(chosen, start=1):
        qid = f"q{i}"
        passages, record = build_one(qid, row)
        if not record["gold_support_ids"]:
            print(f"  WARN {qid}: no gold support found, skipping")
            continue
        corpus.update(passages)
        questions.append(record)

    # --- validation: every gold id must exist in corpus; gold non-empty ---
    problems = []
    for q in questions:
        if not q["gold_support_ids"]:
            problems.append(f"{q['qid']}: empty gold_support_ids")
        for gid in q["gold_support_ids"]:
            if gid not in corpus:
                problems.append(f"{q['qid']}: gold id {gid} missing from corpus")
    if problems:
        raise SystemExit("DATA VALIDATION FAILED:\n  " + "\n  ".join(problems))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "corpus.json"), "w") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "questions.json"), "w") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    tally = {}
    for q in questions:
        tally[(q["type"], q["level"])] = tally.get((q["type"], q["level"]), 0) + 1
    print(f"\nWrote {len(questions)} questions, {len(corpus)} passages.")
    print("Stratification (type, level): "
          + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main()
