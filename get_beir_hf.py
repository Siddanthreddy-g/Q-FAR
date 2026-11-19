"""Download a BEIR dataset from its Hugging Face copy (BeIR/<name>) into datasets/<name>, in the folder format the
other scripts read, for when the BEIR server (public.ukp.informatik.tu-darmstadt.de) does not respond.

pip install datasets
python get_beir_hf.py arguana
Then add  --data_path datasets/arguana  to sa_aqee_soft.py, compare_methods.py and significance.py.
"""
import json
import os
import sys

from datasets import load_dataset

name = sys.argv[1] if len(sys.argv) > 1 else "arguana"
split = sys.argv[2] if len(sys.argv) > 2 else "test"
out = os.path.join("datasets", name)
os.makedirs(os.path.join(out, "qrels"), exist_ok=True)

corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
qrels = load_dataset(f"BeIR/{name}-qrels", split=split)

with open(os.path.join(out, "corpus.jsonl"), "w", encoding="utf-8") as f:
    for r in corpus:
        f.write(json.dumps({"_id": str(r["_id"]), "title": r.get("title") or "", "text": r["text"] or ""}) + "\n")
with open(os.path.join(out, "queries.jsonl"), "w", encoding="utf-8") as f:
    for r in queries:
        f.write(json.dumps({"_id": str(r["_id"]), "text": r["text"] or ""}) + "\n")
with open(os.path.join(out, "qrels", f"{split}.tsv"), "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for r in qrels:
        f.write(f"{r['query-id']}\t{r['corpus-id']}\t{int(r['score'])}\n")
print(f"{name}: {len(corpus):,} documents, {len(queries):,} queries, {len(qrels):,} relevance judgements -> {out}")
