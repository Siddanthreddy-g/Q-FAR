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
The method's default, fixed in advance, is rho = 0.5 with the membership rule; the script also reports a sweep of rho
and both rules, K-Means with the identical spill, and the method without the evolution, so the source of any gain is
visible.

## Datasets
BEIR NQ (a 200,000-document subset, 3,452 queries), SciFact, NFCorpus, and SIFT1M (first 200,000 vectors). Datasets and
embedding caches are not in this repository.

## Results
* At the same number of documents searched, SA-AQEE-FCM is close to K-Means on the text datasets (a little below it on
  NQ and NFCorpus, no consistent difference on SciFact) and worse on SIFT1M. It does not beat K-Means or HNSW.
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
```
`sa_aqee_soft.py --synthetic` runs the whole pipeline on synthetic data as a quick check (those numbers are not
results). `--data_path` loads a BEIR-format folder that is already on disk instead of downloading it.
The scripts expect the BEIR datasets and cached embeddings in the folders they create on first use.

## Layout
```
*.py          experiment code (clustering, retrieval, cross-tests, routing)
cache_cross/  saved result files (JSON)
```
Embedding caches, datasets and the virtual environment are not tracked.

