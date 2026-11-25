# Q-FAR: clustering-based retrieval for RAG

Author: G. Siddanth Reddy

Q-FAR looks at a simple question: if a document collection is split into clusters and each query searches only the
nearest few clusters, how much retrieval quality is lost compared with searching everything, and how does that compare
with a standard partition-based index (FAISS IVF)?

All clustering code (K-Means, Fuzzy C-Means, and SA-AQEE-FCM) is written from scratch in NumPy. Documents and queries
are embedded with `BAAI/bge-small-en-v1.5`.

## Methods compared
| Name in the report | What it is |
|---|---|
| K-Means (previous method) | Hard clustering, `kmean_retrival.py` |
| Fuzzy C-Means (previous method) | Soft clustering with a fixed fuzzifier m, `fuzzycmeans_scratch.py` |
| SA-AQEE-FCM (V5, the original plan) | Quantum-inspired evolutionary search with self-adaptive rotation and an adaptive fuzzifier, `sa_aqee_fcm.py` |
| SA-AQEE-FCM-S (proposed method) | SA-AQEE-FCM with soft assignment of boundary documents, `sa_aqee_soft.py` (see below) |
| FAISS IVF | Standard partition-based index, `their_ann_methods_on_our_data.py` |
| Contriever, DPR, bge-small | Embedding models, `cross_test_their_models.py` |
| Murali and Damodaram (2015) | Our reimplementation of their clustering method, `pfcm_murali_baseline.py` |
| All of the above at equal cost | `compare_methods.py`: exhaustive search, FCM and FAISS IVF next to K-Means, V5, SOAR and the proposed method, on all four metrics |

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
history (commit 7b856cb). NFCorpus has its own development split: ranked on all four metrics (`--criterion all`),
its dev queries chose rho = 0.25, SOAR, lambda = 2 (`cache_cross/dev_select_soft_nfcorpus_dev_all.json`), which the
NFCorpus results use. NQ has no development split in BEIR and uses the SciFact setting.

## Datasets
BEIR NQ (a 200,000-document subset, 3,452 queries), SciFact, NFCorpus, and SIFT1M (first 200,000 vectors). Datasets and
embedding caches are not in this repository.

## Results
### SA-AQEE-FCM-S against K-Means
Each dataset uses the setting chosen on its own development queries with `dev_select_soft.py`: SciFact (its train
queries) rho = 0.5, SOAR, lambda = 0.5; NFCorpus (its dev queries, ranked on all four metrics) rho = 0.25, SOAR,
lambda = 2; NQ has no development split in BEIR and uses the SciFact setting, as does SIFT1M. Compared with K-Means at
the same number of documents searched; mean of seeds 42, 43 and 44; K-Means runs 50 iterations and is scored with the
better of its two routings. Points where K-Means has no result at a similar cost are left out.

| Dataset | clusters searched | docs searched | NDCG@10 proposed | NDCG@10 K-Means | difference | Recall@100 difference | index |
|---|---|---|---|---|---|---|---|
| SciFact | 1 | 472 | 0.5524 | 0.5381 | +0.0143 | +0.0192 | 1.50x |
| SciFact | 2 | 943 | 0.6710 | 0.6566 | +0.0144 | +0.0257 | 1.50x |
| SciFact | 3 | 1,420 | 0.7000 | 0.6864 | +0.0136 | +0.0197 | 1.50x |
| SciFact | 4 | 1,892 | 0.7065 | 0.7007 | +0.0058 | +0.0120 | 1.50x |
| NQ (200k) | 1 | 4,899 | 0.5729 | 0.5553 | +0.0176 | +0.0190 | 1.50x |
| NQ (200k) | 2 | 9,806 | 0.6690 | 0.6676 | +0.0014 | -0.0003 | 1.50x |
| NQ (200k) | 3 | 14,637 | 0.7036 | 0.6965 | +0.0071 | +0.0081 | 1.50x |
| NQ (200k) | 4 | 19,397 | 0.7194 | 0.7128 | +0.0066 | +0.0072 | 1.50x |
| NFCorpus | 1 | 80 | 0.2117 | 0.2051 | +0.0066 | +0.0153 | 1.25x |
| NFCorpus | 2 | 162 | 0.2426 | 0.2428 | -0.0002 | +0.0054 | 1.25x |
| NFCorpus | 3 | 245 | 0.2629 | 0.2604 | +0.0024 | +0.0129 | 1.25x |
| NFCorpus | 4 | 328 | 0.2800 | 0.2755 | +0.0045 | +0.0108 | 1.25x |

SIFT1M (first 200,000 vectors, seed 42 only), overlap of the top 10 with the exact top 10 at the same number of vectors
searched: 0.693 vs 0.667 (about 5,000 vectors), 0.867 vs 0.841 (about 10,000) and 0.966 vs 0.949 (about 20,000). The
share of queries whose exact nearest neighbour comes first is also higher: 0.745 vs 0.721, 0.899 vs 0.881 and 0.978 vs
0.968.

* Across all four metrics (NDCG, MAP, Recall and Precision at cutoffs 1, 3, 5, 10 and 100, 1 to 4 clusters
  searched), SA-AQEE-FCM-S is ahead of K-Means in 80 of 80 comparisons on SciFact, 76 of 80 on NQ (the 4 misses are
  all at 2 clusters and under 0.0005) and 66 of 80 on NFCorpus. On NFCorpus every MAP value and every value at
  cutoffs 1 and 100 is ahead; the misses are at cutoffs 3 to 10, mostly at 2 clusters, the largest -0.0056 (P@5),
  inside the seed-to-seed spread there (NDCG@10 standard deviation up to 0.009).
* On NDCG@10 alone it is ahead at every compared point on SciFact (+0.006 to +0.014) and NQ (+0.001 to +0.018), and
  on NFCorpus at 1, 3 and 4 clusters (+0.002 to +0.007) with a tie at 2 (-0.0002).
* What the parts contribute (ablations from the same runs, at each dataset's setting):
  * Soft assignment: K-Means' own clusters with the same soft assignment also gain over plain K-Means. On NQ they are
    ahead of SA-AQEE-FCM-S by 0.003 to 0.004 at 2 to 4 clusters; SA-AQEE-FCM-S is ahead of them on SciFact by 0.008
    to 0.022 at 2 to 4 clusters, and on NFCorpus by 0.002 to 0.005 at 3 and 4 clusters (0.001 behind at 2).
  * Evolution: its effect is small and not consistent. With it, NDCG@10 is higher by 0.001 to 0.004 on NQ and by
    0.002 to 0.007 on SciFact at 2 to 4 clusters, but at 1 cluster on SciFact the method without it is ahead by 0.025;
    on NFCorpus it is 0.002 to 0.004 lower at 2 and 3 clusters and level at 1 and 4.
  * Spherical FCM without soft assignment is a little below K-Means on NQ and NFCorpus.
* Earlier runs are kept: the SciFact setting on NFCorpus (51 of 80 four-metric comparisons ahead) is
  `cache_cross/sa_aqee_soft_nfcorpus_scifact_setting.json`, and the first default (membership rule, fixed before any
  run) is in the git history (commit 7b856cb).
* Giving every document a second copy (2x index) did worse than these settings on NQ and NFCorpus.

### Every method at equal cost
`compare_methods.py` runs exhaustive search, Fuzzy C-Means (m = 2) and FAISS IVF on the same vectors, queries and
evaluation and reads each method off its own best-quality-per-cost curve at the proposed method's costs. NQ-200k at 4,899 documents compared per query (the proposed method at 1 cluster):

| Method | NDCG@10 | MAP@10 | Recall@100 | P@10 |
|---|---|---|---|---|
| Exhaustive search (200,000 documents) | 0.7556 | 0.7032 | 0.9711 | 0.1080 |
| K-Means | 0.5553 | 0.5188 | 0.6833 | 0.0770 |
| Fuzzy C-Means (m = 2) | cannot reach this cost | | | |
| FAISS IVF | 0.5394 | 0.5047 | 0.6589 | 0.0747 |
| SA-AQEE-FCM V5 (seeds 42, 43) | 0.5427 | 0.5076 | 0.6651 | 0.0751 |
| K-Means + learned router (seed 42) | 0.5657 | 0.5294 | 0.6884 | 0.0777 |
| SA-AQEE-FCM-S (proposed) | 0.5729 | 0.5358 | 0.7023 | 0.0794 |

* Fuzzy C-Means collapses on all three text collections: its memberships equal the flat 1/C, and its cheapest setting
  (1 cluster) already compares 35,476 documents on NQ (NDCG@10 0.357), 1,478 on SciFact and 317 on NFCorpus.
* Over all four metrics at 5 cutoffs and 1 to 4 clusters, the proposed method is ahead of FAISS IVF in 80 of 80
  comparisons on NQ, 78 of 80 on SciFact and 73 of 80 on NFCorpus.
* Seed 44 of SA-AQEE-FCM V5 on NQ put about 95,000 documents in one cluster, so the V5 row uses seeds 42 and 43.

### Significance (`significance.py`)
Paired bootstrap over queries (10,000 resamples, 95% confidence intervals), Holm's correction over each method's four
budgets; the re-run reproduces the means above to within 0.00001. NDCG@10 of the proposed method minus each method:
* K-Means: significant on NQ (3,452 queries) at 1, 3 and 4 clusters, +0.018 [+0.010, +0.026] at 1 cluster; positive but
  not significant on SciFact (300 queries) and NFCorpus (323).
* FAISS IVF: significant on NQ at all four budgets (+0.011 to +0.034); not significant on the small collections.
* The same partition with randomly chosen border documents (same number of copies): the fuzzy gate is significantly
  better at all four budgets on NQ (+0.012 to +0.024), at 2 to 4 clusters on SciFact and at 4 on NFCorpus.
* SOAR (2x index): significantly better at 2 clusters on NQ (+0.014), no significant difference elsewhere.
* K-Means with the same soft assignment: no significant difference on NQ or NFCorpus; the proposed method is
  significantly better on SciFact at 2 to 4 clusters.
* Without the evolution: no significant difference, except SciFact at 1 cluster, where the version without it is
  better (-0.024).
* NQ index sizes: K-Means and FAISS IVF 295 MB, proposed 442 MB, SOAR 589 MB.

### FiQA: a new collection with the settings unchanged
FiQA (57,638 documents, 648 queries, C = 64) was added after every setting had been frozen and run once with the
SciFact setting (rho 0.5, SOAR, lambda 0.5; index 1.50x). NDCG@10 of the proposed method minus each method, at 1, 2, 3
and 4 clusters (1,495, 3,013, 4,505 and 5,982 documents compared):
* K-Means: +0.019, +0.015, +0.010, +0.009, significant at all four budgets (+0.019 [+0.008, +0.030] at 1 cluster);
  ahead in 80 of 80 four-metric comparisons.
* SOAR (2x index): +0.020, +0.010, +0.009 at 2 to 4 clusters, significant at all three; index 128 MB against 170 MB.
* Randomly chosen border documents: +0.008, +0.010, +0.008, +0.005, significant at all four budgets.
* FAISS IVF (+0.006 to +0.009) and K-Means with the same soft assignment (+0.004 to +0.009): ahead at every budget,
  not significant after Holm's correction.
* Without the evolution: +0.010 at 2 clusters (significant), +0.004 and +0.002 at 3 and 4 (not significant).

### Earlier results
* SA-AQEE-FCM (V5) is close to K-Means on the text datasets (a little below it on NQ and NFCorpus, no consistent
  difference on SciFact) and worse on SIFT1M. It does not beat K-Means. In the new NQ runs, one seed (44,
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
python dev_select_soft.py --dataset nfcorpus --split dev --ks 64 --criterion all
python sa_aqee_soft.py --dataset nfcorpus --k 64 --seeds 42 43 44 --default_rho 0.25 --default_rule soar --soar_lambda 2
python sa_aqee_soft.py --dataset nq --subsample 200000 --emb_dir cache --k 64 --seeds 42 43 44 --v5_fit_sample 30000 --m_min 1.02
python our_model_on_sift1m.py --sa_aqee_soft
python dev_select_soft.py
python compare_methods.py --dataset scifact
python compare_methods.py --dataset nfcorpus --k 64
python compare_methods.py --dataset nq --subsample 200000 --emb_dir cache --k 64
python significance.py --dataset scifact
python significance.py --dataset nfcorpus --k 64
python significance.py --dataset nq --subsample 200000 --emb_dir cache --k 64
python sa_aqee_soft.py --dataset fiqa --k 64 --seeds 42 43 44
python compare_methods.py --dataset fiqa --k 64
python significance.py --dataset fiqa --k 64
```
`significance.py` re-runs the compared methods with the same seeds, keeps every query's scores and gives each
difference a paired-bootstrap 95% confidence interval, bootstrap and Wilcoxon p-values and Holm's correction. It
also tests the proposed method against the same partition with randomly chosen border documents, and reports index
size and build time.
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

