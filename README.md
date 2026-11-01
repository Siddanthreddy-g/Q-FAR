# Q-FAR: clustering-based retrieval for RAG

Author: G. Siddanth Reddy

Q-FAR looks at a simple question: if a document collection is split into clusters and each query searches only the
nearest few clusters, how much retrieval quality is lost compared with searching everything, and how does that compare
with standard approximate-search indexes (FAISS IVF and HNSW)?

All clustering code (K-Means, Fuzzy C-Means, and SA-AQEE-FCM) is written from scratch in NumPy. Documents and queries
are embedded with `BAAI/bge-small-en-v1.5`.

## Methods compared
| Name in the report | What it is |
|---|---|
| K-Means (previous method) | Hard clustering, `kmean_retrival.py` |
| Fuzzy C-Means (previous method) | Soft clustering with a fixed fuzzifier m, `fuzzycmeans_scratch.py` |
| SA-AQEE-FCM (ours) | Quantum-inspired evolutionary search with self-adaptive rotation and an adaptive fuzzifier, `sa_aqee_fcm.py` |
| SA-AQEE-FCM-S (ours) | SA-AQEE-FCM with soft assignment of boundary documents, `sa_aqee_soft.py` (see below) |
| FAISS IVF, HNSW | Standard indexes, `their_ann_methods_on_our_data.py` |
| Contriever, DPR, bge-small | Embedding models, `cross_test_their_models.py` |
| Murali and Damodaram (2015) | Our reimplementation of their clustering method, `pfcm_murali_baseline.py` |

### SA-AQEE-FCM-S
In SA-AQEE-FCM the fuzzifier settles at its floor (m* = 1.02), so the final partition is effectively K-Means, and the
document memberships are discarded (each document keeps one cluster). SA-AQEE-FCM-S uses the fuzziness where retrieval
can gain from it:
* Spherical FCM fitted on the whole corpus; queries are routed by inner product to the unit centroids.
* The quantum-inspired evolution searches data-anchored seeds (an index into a k-means++ candidate pool for each
  cluster, 512 qubits for 64 clusters). Its fitness is label-free: the share of each pseudo-query's exact neighbours
  that lie in the clusters it would search at a fixed number of documents searched.
* The self-adaptive fuzzifier sets how fuzzy the document assignment is. m is adapted with an Eq. (9)-style rule until
  the share of documents whose second membership is at least half their first equals a budget rho. Those boundary
  documents also get a copy in their second cluster ("membership"), or in the cluster chosen by the SOAR loss of
  Sun et al. (NeurIPS 2023) ("soar").

Copies count toward "documents searched", and the index size (x1.5 for rho = 0.5) is reported with every result.
The script also reports a sweep of rho and both rules, K-Means with the identical spill, and the method without the
evolution, so the source of any gain is visible.

**Default setting: rho = 0.5, SOAR rule, lambda = 0.5.** It was chosen on development queries only, with
`dev_select_soft.py`: SciFact's train queries (809, used by no reported result) embedded with all-MiniLM-L6-v2, compared
with K-Means at 17 and 64 clusters over seeds 42 to 44. Stage 1 compared the spill budget and the rule (SOAR was ahead
at every budget); stage 2 compared lambda in {0.25, 0.5, 1, 2, 4} (0.5 was best, 2 and 4 were clearly worse). Routing
each query by several prototypes per cluster was also tried on the development queries and did not help, so it is not
part of the method. The first default (rho = 0.5, membership rule) was fixed before any run; the Results below still
use it.

## Datasets
BEIR NQ (a 200,000-document subset, 3,452 queries), SciFact, NFCorpus, and SIFT1M (first 200,000 vectors). Datasets and
embedding caches are not in this repository.

## Results
### SA-AQEE-FCM-S against K-Means
These runs use the first default (membership rule); the current default (SOAR rule, lambda = 0.5) has not been run on
these datasets with bge-small yet. At the same number of documents searched, with the setting fixed before the runs (half of the documents get a second
copy, so the index holds 1.5x the documents). Mean of seeds 42, 43 and 44; K-Means runs 50 iterations and is scored
with the better of its two routings. Points where K-Means has no result at a similar cost are left out.

| Dataset | clusters searched | docs searched | NDCG@10 ours | NDCG@10 K-Means | difference | Recall@100 difference |
|---|---|---|---|---|---|---|
| SciFact | 1 | 476 | 0.5579 | 0.5392 | +0.0187 | +0.0215 |
| SciFact | 2 | 957 | 0.6725 | 0.6590 | +0.0136 | +0.0179 |
| SciFact | 3 | 1,446 | 0.6976 | 0.6871 | +0.0105 | +0.0157 |
| SciFact | 4 | 1,923 | 0.7060 | 0.7017 | +0.0044 | +0.0087 |
| NQ (200k) | 1 | 4,969 | 0.5755 | 0.5578 | +0.0177 | +0.0196 |
| NQ (200k) | 2 | 10,042 | 0.6671 | 0.6710 | -0.0039 | -0.0061 |
| NQ (200k) | 3 | 15,033 | 0.6996 | 0.6979 | +0.0018 | +0.0020 |
| NQ (200k) | 4 | 19,925 | 0.7155 | 0.7139 | +0.0016 | +0.0013 |
| NFCorpus | 1 | 100 | 0.2170 | 0.2155 | +0.0014 | +0.0110 |
| NFCorpus | 2 | 210 | 0.2484 | 0.2537 | -0.0052 | +0.0023 |
| NFCorpus | 3 | 320 | 0.2705 | 0.2745 | -0.0039 | +0.0082 |
| NFCorpus | 4 | 429 | 0.2864 | 0.2919 | -0.0055 | +0.0059 |

SIFT1M (first 200,000 vectors, seed 42 only), overlap of the top 10 with the exact top 10 at the same number of vectors
searched: 0.689 vs 0.669 (about 5,000 vectors), 0.862 vs 0.844 (about 10,000) and 0.964 vs 0.951 (about 20,000).

* SA-AQEE-FCM-S is above K-Means at every compared point on SciFact and SIFT1M, and on NQ when one cluster is searched
  (+0.018). On NQ with 2 to 4 clusters the two are within 0.004 of each other. On NFCorpus it is below K-Means on
  NDCG@10 at 2 to 4 clusters (by 0.004 to 0.006), although its Recall@100 is higher there. The seed-to-seed standard
  deviation of NDCG@10 is 0.002 to 0.011.
* What the parts contribute (ablations from the same runs):
  * Soft assignment: on NQ, K-Means' own clusters with the same soft assignment do better than SA-AQEE-FCM-S (it is
    below them at 29 of 32 points, by 0.005 on average). On SciFact the order is reversed: SA-AQEE-FCM-S is above them by
    0.004 on average and by 0.009 to 0.020 at the default setting. On NFCorpus neither is consistently ahead.
  * Evolution: with and without it, NDCG@10 differs by about 0.001 on average (slightly higher with it on NQ and
    SciFact, slightly lower on NFCorpus).
  * Spherical FCM without soft assignment is a little below K-Means on NQ and NFCorpus (by 0.003 to 0.004 on average).
* Other settings in the sweep (not chosen in advance, so not the headline): placing the second copy with the SOAR rule
  did better than the membership rule on NQ, NFCorpus and SIFT1M and about the same on SciFact; giving every document a
  second copy (2x index) was worse than K-Means on NQ, NFCorpus and SIFT1M.
* HNSW was not re-run for SA-AQEE-FCM-S.

### Earlier results
* SA-AQEE-FCM (V5) is close to K-Means on the text datasets (a little below it on NQ and NFCorpus, no consistent
  difference on SciFact) and worse on SIFT1M. It does not beat K-Means or HNSW. In the new NQ runs, one seed (44,
  m* = 1.105) produced a cluster so large that searching one cluster scanned about 95,000 documents on average.
* Adaptive and learned routing improved both clusterings a little and did not change their ranking.

The raw numbers are in `cache_cross/` (JSON).

## Running the experiments
```
pip install -r requirements.txt
python sa_aqee_fcm.py --dataset nfcorpus --seeds 42 43 44
python sa_aqee_fcm.py --dataset nq --subsample 200000 --emb_dir cache --variants V5 --k 64 --m_min 1.02 --fit_sample 30000 --n_search 1 2 4 8 16 25
python sa_aqee_routing.py --dataset scifact --seeds 42 43 44
python sa_aqee_learned_routing.py --dataset scifact --seeds 42 43 44 --oracle
python our_model_on_sift1m.py --sa_aqee
python sa_aqee_soft.py --dataset scifact --seeds 42 43 44
python sa_aqee_soft.py --dataset nfcorpus --k 64 --seeds 42 43 44
python sa_aqee_soft.py --dataset nq --subsample 200000 --emb_dir cache --k 64 --seeds 42 43 44 --v5_fit_sample 30000 --m_min 1.02
python our_model_on_sift1m.py --sa_aqee_soft
python dev_select_soft.py
```
`dev_select_soft.py` repeats the choice of the default setting on SciFact's train queries (it embeds them with
all-MiniLM-L6-v2 through sentence-transformers). `sa_aqee_soft.py --synthetic` runs the whole pipeline on synthetic data as a quick check (those numbers are not
results). `--data_path` loads a BEIR-format folder that is already on disk instead of downloading it.
The scripts expect the BEIR datasets and cached embeddings in the folders they create on first use.

## Layout
```
*.py          experiment code (clustering, retrieval, cross-tests, routing)
cache_cross/  saved result files (JSON)
```
Embedding caches, datasets and the virtual environment are not tracked.

