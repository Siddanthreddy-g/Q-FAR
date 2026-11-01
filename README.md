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
part of the method. The selection was repeated with
sentence-transformers on a second machine and gave the same choice and the same numbers; the record is
`cache_cross/dev_select_soft_scifact_train.json`. The first default (rho = 0.5, membership rule) was fixed before any run; its results are in the git
history (commit 7b856cb), and the Results below use the current default.

## Datasets
BEIR NQ (a 200,000-document subset, 3,452 queries), SciFact, NFCorpus, and SIFT1M (first 200,000 vectors). Datasets and
embedding caches are not in this repository.

## Results
### SA-AQEE-FCM-S against K-Means
Default setting (rho = 0.5, SOAR rule, lambda = 0.5; the index holds 1.5x the documents), compared with K-Means at the
same number of documents searched. Mean of seeds 42, 43 and 44; K-Means runs 50 iterations and is scored with the
better of its two routings. Points where K-Means has no result at a similar cost are left out.

| Dataset | clusters searched | docs searched | NDCG@10 ours | NDCG@10 K-Means | difference | Recall@100 difference |
|---|---|---|---|---|---|---|
| SciFact | 1 | 472 | 0.5524 | 0.5381 | +0.0143 | +0.0192 |
| SciFact | 2 | 943 | 0.6710 | 0.6566 | +0.0144 | +0.0257 |
| SciFact | 3 | 1,420 | 0.7000 | 0.6864 | +0.0136 | +0.0197 |
| SciFact | 4 | 1,892 | 0.7065 | 0.7007 | +0.0058 | +0.0120 |
| NQ (200k) | 1 | 4,899 | 0.5729 | 0.5553 | +0.0176 | +0.0190 |
| NQ (200k) | 2 | 9,806 | 0.6690 | 0.6676 | +0.0014 | -0.0003 |
| NQ (200k) | 3 | 14,637 | 0.7036 | 0.6965 | +0.0071 | +0.0081 |
| NQ (200k) | 4 | 19,397 | 0.7194 | 0.7128 | +0.0066 | +0.0072 |
| NFCorpus | 1 | 99 | 0.2211 | 0.2155 | +0.0057 | +0.0154 |
| NFCorpus | 2 | 205 | 0.2526 | 0.2528 | -0.0001 | +0.0049 |
| NFCorpus | 3 | 311 | 0.2743 | 0.2734 | +0.0009 | +0.0125 |
| NFCorpus | 4 | 416 | 0.2894 | 0.2911 | -0.0017 | +0.0079 |

SIFT1M (first 200,000 vectors, seed 42 only), overlap of the top 10 with the exact top 10 at the same number of vectors
searched: 0.693 vs 0.667 (about 5,000 vectors), 0.867 vs 0.841 (about 10,000) and 0.966 vs 0.949 (about 20,000).

* SA-AQEE-FCM-S is above K-Means at every compared point on SciFact (+0.006 to +0.014), NQ (+0.001 to +0.018) and
  SIFT1M (+0.016 to +0.026). On NFCorpus it is above K-Means at 1 and 3 clusters and below it by at most 0.002 at 2 and
  4 clusters, which is inside the seed-to-seed spread (standard deviation 0.003 to 0.007); its Recall@100 is higher at
  every NFCorpus point.
* What the parts contribute (ablations from the same runs, at the default setting):
  * Soft assignment: K-Means' own clusters with the same soft assignment also gain over plain K-Means. On NQ they are
    ahead of SA-AQEE-FCM-S by 0.003 to 0.004 at 2 to 4 clusters; on SciFact SA-AQEE-FCM-S is ahead of them by 0.008
    to 0.022 at 2 to 4 clusters; on NFCorpus neither is consistently ahead; on SIFT1M SA-AQEE-FCM-S has the higher
    overlap and they have the higher NN@1.
  * Evolution: with it, NDCG@10 is higher by 0.001 to 0.007 on SciFact and NQ and within 0.002 either way on NFCorpus.
  * Spherical FCM without soft assignment is a little below K-Means on NQ and NFCorpus.
* The first default (membership rule, fixed before any run) was below K-Means on NFCorpus by up to 0.006 and at one
  NQ point; those runs are in the git history (commit 7b856cb).
* Giving every document a second copy (2x index) did not do better than the default on NQ or NFCorpus.
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

