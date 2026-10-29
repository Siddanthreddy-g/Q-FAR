import argparse
import json
import os
import random
import time

import numpy as np


# ---------------------------------------------------------------------------
# Data: identical subsample logic to kmeans_retrieval.py (same seed, same order)
# ---------------------------------------------------------------------------
def load_data(dataset="nq"):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split="test")
    print(f"Loaded corpus={len(corpus):,} queries={len(queries):,}")
    return corpus, queries, qrels


def subsample_corpus(corpus, qrels, target_size):
    if target_size is None or target_size >= len(corpus):
        return corpus
    random.seed(42)                                   # same state as kmeans_retrieval.py at this point
    relevant = set()
    for docs in qrels.values():
        relevant.update(docs.keys())
    all_ids = list(corpus.keys())
    extra_needed = max(0, target_size - len(relevant))
    pool = [d for d in all_ids if d not in relevant]
    extra = random.sample(pool, min(extra_needed, len(pool)))
    keep = relevant | set(extra)
    keep_ids = [d for d in all_ids if d in keep]      # filter the stable list, never iterate a set
    print(f"Subsampled corpus: {len(keep_ids):,} docs ({len(relevant):,} relevant kept)")
    return {d: corpus[d] for d in keep_ids}


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def _batches(order, batch_size):
    for s in range(0, len(order), batch_size):
        yield order[s:s + batch_size]


def encode_hf(texts, tokenizer, model, pooling, batch_size, max_length, pairs=None, desc="encoding"):
    """Generic BERT-style encoder. pooling = 'mean' (Contriever) or 'pooler' (DPR).
    pairs: optional second text per item (DPR passages are encoded as title [SEP] text)."""
    import torch
    from tqdm import tqdm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([-len(t) for t in texts])      # long texts first -> less padding waste
    with torch.no_grad():
        for idx in tqdm(list(_batches(order, batch_size)), desc=desc):
            a = [texts[i] for i in idx]
            if pairs is not None:
                enc = tokenizer(a, [pairs[i] for i in idx], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            else:
                enc = tokenizer(a, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            res = model(**enc)
            if pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).type_as(res.last_hidden_state)
                emb = (res.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                emb = res.pooler_output
            out[idx] = emb.cpu().numpy()
    return out


def embed(model_name, doc_titles, doc_texts, query_texts, batch_size, max_length):
    from transformers import AutoModel, AutoTokenizer
    if model_name == "contriever":
        tok = AutoTokenizer.from_pretrained("facebook/contriever")
        mdl = AutoModel.from_pretrained("facebook/contriever")
        joined = [(t + " " + x).strip() for t, x in zip(doc_titles, doc_texts)]
        d = encode_hf(joined, tok, mdl, "mean", batch_size, max_length, desc="Contriever docs")
        q = encode_hf(query_texts, tok, mdl, "mean", batch_size, 64, desc="Contriever queries")
        return d, q
    if model_name == "dpr":
        from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                                  DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
        ctok = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        cmdl = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        qtok = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        qmdl = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        d = encode_hf(doc_titles, ctok, cmdl, "pooler", batch_size, max_length,
                      pairs=doc_texts, desc="DPR passages")
        q = encode_hf(query_texts, qtok, qmdl, "pooler", batch_size, 64, desc="DPR questions")
        return d, q
    if model_name == "bge":
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("BAAI/bge-small-en-v1.5")
        joined = [(t + " " + x).strip() for t, x in zip(doc_titles, doc_texts)]
        d = st.encode(joined, batch_size=256, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True)
        q = st.encode(query_texts, batch_size=256, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True)
        return d, q
    raise ValueError(model_name)


# ---------------------------------------------------------------------------
# Search + metrics
# ---------------------------------------------------------------------------
def flat_search(doc_emb, q_emb, doc_ids, query_ids, top_k=100):
    import faiss
    index = faiss.IndexFlatIP(doc_emb.shape[1])          # dot product, as in both papers
    index.add(np.ascontiguousarray(doc_emb, dtype="float32"))
    t0 = time.time()
    scores, ids = index.search(np.ascontiguousarray(q_emb, dtype="float32"), top_k)
    ms = (time.time() - t0) / len(query_ids) * 1000
    results = {query_ids[i]: {doc_ids[j]: float(s) for j, s in zip(ids[i], scores[i])}
               for i in range(len(query_ids))}
    return results, ms


def success_at_k(results, qrels, ks=(5, 20, 100)):
    """Top-k accuracy in the spirit of the DPR paper (hit if any judged-relevant doc is in the top k)."""
    out = {}
    for k in ks:
        hits = 0
        for qid, docs in results.items():
            rel = {d for d, s in qrels.get(qid, {}).items() if s > 0}
            top = sorted(docs, key=docs.get, reverse=True)[:k]
            hits += int(any(d in rel for d in top))
        out[f"Success@{k}"] = hits / len(results)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["contriever", "dpr", "bge"], required=True)
    ap.add_argument("--target_size", type=int, default=200000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    corpus, queries, qrels = load_data("nq")
    corpus = subsample_corpus(corpus, qrels, args.target_size)
    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    titles = [corpus[d].get("title", "") for d in doc_ids]
    texts = [corpus[d]["text"] for d in doc_ids]
    qtexts = [queries[q] for q in query_ids]

    os.makedirs("cache_cross", exist_ok=True)
    tag = f"{args.model}_{len(doc_ids)}"
    dpath, qpath = f"cache_cross/{tag}_docs.npy", f"cache_cross/{tag}_queries.npy"

    # Re-use your existing bge-small cache when it matches the 200K subsample
    if (args.model == "bge" and len(doc_ids) == 200000
            and os.path.exists("cache/corpus_emb.npy") and os.path.exists("cache/query_emb.npy")):
        print("Re-using your existing bge-small embeddings from cache/")
        d_emb, q_emb = np.load("cache/corpus_emb.npy"), np.load("cache/query_emb.npy")
    elif os.path.exists(dpath) and os.path.exists(qpath):
        print("Loading cached embeddings", tag)
        d_emb, q_emb = np.load(dpath), np.load(qpath)
    else:
        t0 = time.time()
        d_emb, q_emb = embed(args.model, titles, texts, qtexts, args.batch_size, args.max_length)
        print(f"Embedding took {(time.time() - t0) / 60:.1f} min")
        np.save(dpath, d_emb)
        np.save(qpath, q_emb)

    assert len(d_emb) == len(doc_ids), "embedding count != corpus size (stale cache? delete cache_cross/)"

    from beir.retrieval.evaluation import EvaluateRetrieval
    results, ms = flat_search(d_emb, q_emb, doc_ids, query_ids)
    ndcg, _map, recall, precision = EvaluateRetrieval().evaluate(qrels, results, [1, 3, 5, 10, 100])
    succ = success_at_k(results, qrels)

    print(f"\n=== {args.model} | flat search | NQ subsample {len(doc_ids):,} docs ===")
    print(f"avg search latency / query: {ms:.2f} ms (flat, exhaustive)")
    print("NDCG:", ndcg)
    print("Recall:", recall)
    print("Precision:", precision)
    print("MAP:", _map)
    print("DPR-style top-k accuracy:", succ)

    with open(f"cache_cross/results_{tag}.json", "w") as f:
        json.dump({"model": args.model, "docs": len(doc_ids), "latency_ms": ms, "ndcg": ndcg,
                   "recall": recall, "precision": precision, "map": _map, "success": succ}, f, indent=2)


if __name__ == "__main__":
    main()
