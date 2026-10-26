# Experiment 001: Replace the network with a kNN vote over embeddings

- Status: Concluded. Keep the network.
- Date: October 2026
- Model compared: `1.25.5-f146986` (the production model at the time)
- Training data: `raw_source_data.zip` in the model bucket, dated 2025-09-17
- Benchmark data: `benchmarking_data.zip` in the model bucket, dated 2024-06-10

## Question

The sentence transformer (`all-mpnet-base-v2`) is frozen. The network only
classifies its embeddings. Can a nearest-neighbour vote over the training
embeddings replace the network?

## Method

`train.py --knn-index-only` uses the normal training data pipeline (sources,
cleaning, multipliers, authoritative overrides). It stores one fp16 embedding
for each unique description, with the count of each code for that description.
The code order matches the production model's `subheadings.json`.

`KNNClassifier` embeds the query and takes the `k` most similar descriptions.
Each neighbour votes `exp((sim - max_sim) / T) * P(code | description)`.
`P(code | description)` comes from the stored counts. The classifier adds the
votes for each code at the requested number of digits, normalises them, and
applies the same result cutoffs as `FlatClassifier`. `T` was 0.05 for all runs
and was not tuned. The runs compared `k` = 10, 25 and 50.

Both classifiers used the same benchmark sets and `benchmark.py` settings as
the automated model build. The network results reproduced the published
results for this model exactly, item by item.

## Results

Percentages of benchmark items. "Heading" and "subheading" mean the first
result matched to 4 and 6 digits.

| Set | Classifier | Top 1 | Top 5 | Heading | Subheading |
| --- | --- | --- | --- | --- | --- |
| Good goods descriptions (393) | Network | **59.3** | **86.5** | **87.0** | **65.6** |
| | kNN k=10 | 46.3 | 73.0 | 77.6 | 53.7 |
| | kNN k=25 | 49.4 | 79.4 | 80.2 | 56.7 |
| | kNN k=50 | 48.9 | 79.9 | 81.9 | 57.0 |
| Classified data (1,783) | Network | **45.4** | **69.5** | **66.1** | **51.5** |
| | kNN k=10 | 39.9 | 62.9 | 61.7 | 47.1 |
| | kNN k=25 | 42.0 | 65.3 | 63.2 | 48.9 |
| | kNN k=50 | 43.0 | 67.4 | 63.2 | 49.6 |

Few benchmark descriptions are also training descriptions (14 of 393 and 169
of 1,783). On the other items, the network still leads on top 1 (58.8% against
49.1%, and 43.9% against 41.5%).

The classifiers make different errors. The first result was correct for at
least one of the two classifiers on 65.1% and 52.7% of items. A combination
can gain at most 6 to 7 points of top-1 accuracy.

## Conclusion

Keep the network. kNN is less accurate on both sets. The kNN index is also too
large to deploy: 7,553,835 unique descriptions use 11.6 GB in fp16. The network
is 357 MB. A smaller approximate index would probably lose more accuracy.

kNN has two possible uses. It shows the training descriptions behind a
prediction, which helps to find bad training data. It can also add a correction
without retraining. Neither use needs kNN to serve production traffic.

## Reproduce

Put the bucket's `raw_source_data`, `benchmarking_data` and model `target/`
files in the project root. Then run:

```sh
python train.py --config search-config.toml --knn-index-only
python benchmark.py --classifier knn --knn-k 50 --knn-device cuda \
  --benchmark-goods-descriptions --output json
python benchmark.py --classifier knn --knn-k 50 --knn-device cuda \
  --benchmark-classifieds --number-of-items 100000 --output json
```

The index build needs about 14 GB of memory to load. On an RTX 5090 it takes
about 25 minutes, most of which is embedding the descriptions.
