# Experiment 002: Filter tradeset rows by semantic distance to self-texts

- Status: Concluded. Do not use self-text distance to filter training data.
- Date: October 2026
- Related: [RFC-002](../rfcs/002_semantic_distance_filtering.md), epic HMRC-1966
- Data: the same training, benchmark and model inputs as
  [experiment 001](001_knn_embeddings_classifier.md)

## Question

RFC-002 proposes to remove tradeset rows whose description is far from the
self-text of the labelled code. Does this remove wrong rows, and does the
network get better when it trains without them?

## Method

The reference for each code is the English self-text in
`CN2026_SelfText_EN_DE_FR.csv`, after negation cleaning. All 9,791 codes have
one. The descriptions use the cached `all-mpnet-base-v2` embeddings from the
kNN index. Each unique (description, code) pair from a tradeset file gets
three scores:

- `sim`: cosine similarity to the labelled code's self-text, as in RFC-002.
- `code rank`: the rank of the labelled code among all self-texts.
- `chapter rank`: the rank of the labelled chapter when each chapter scores
  the similarity of its best self-text. Rank 1 means that the closest self-text
  is in the labelled chapter.

Two reference sets calibrate the scores. The benchmark pairs (2,166) have
trusted codes. The known incorrect pairs in `incorrect_code_desc_pairs.csv`
have only 12 rows with a current code.

## Analysis

Trusted benchmark pairs score only a little better than tradeset rows. For
example, the labelled chapter ranks first for 56.0% of benchmark pairs and
44.9% of tradeset rows. Every rule removes many trusted pairs:

| Rule | Tradeset rows removed | Benchmark pairs removed | Known incorrect pairs removed |
| --- | --- | --- | --- |
| `sim` < 0.2 | 22.7% | 13.2% | 5 of 12 |
| `chapter rank` > 3 | 30.6% | 22.1% | 11 of 12 |
| `chapter rank` > 10 | 15.8% | 10.6% | 7 of 12 |

An AI assistant reviewed a random sample of 40 tradeset rows (weighted by row count)
for each chapter rank bucket. It marked a row correct when the code was
plausible at least to the chapter.

| Chapter rank | Plausible | Clearly wrong | Too vague to judge |
| --- | --- | --- | --- |
| 1 | about 38 | about 1 | about 1 |
| 4 to 10 | about 27 | about 5 | about 8 |
| More than 10 | about 24 | about 5 | about 11 |

In the low buckets, the self-text match is usually the error, not the code.
The match fails on brand names ("ibanez grx70qal" is a guitar), short trade
terms and non-English descriptions. These are the descriptions that parcel
data contains, so a filter on this signal removes useful examples.

This review does not check 8-digit accuracy. The signal cannot check it either,
because self-texts of sibling codes are very similar.

## Results

The network was trained with the production settings on the cached embeddings,
once for each rule. Seeds 0 and 1 of the unfiltered baseline show the run to
run variation. Values are percentages of benchmark items.

| Set | Training data | Top 1 | Top 5 | Heading | Subheading |
| --- | --- | --- | --- | --- | --- |
| Good goods descriptions (393) | Production model | 59.3 | 86.5 | 87.0 | 65.6 |
| | Baseline, seed 0 | 57.5 | 85.8 | 85.8 | 64.4 |
| | Baseline, seed 1 | 59.0 | 85.8 | 87.0 | 65.9 |
| | `chapter rank` > 10 removed | 60.3 | 86.8 | 88.0 | 67.2 |
| | `chapter rank` > 3 removed | 59.8 | 84.7 | 86.8 | 66.9 |
| | `sim` < 0.2 removed | 57.8 | 84.7 | 86.0 | 64.6 |
| Classified data (1,783) | Production model | 45.4 | 69.5 | 66.1 | 51.5 |
| | Baseline, seed 0 | 44.7 | 69.3 | 66.1 | 51.1 |
| | Baseline, seed 1 | 45.3 | 69.0 | 65.0 | 51.5 |
| | `chapter rank` > 10 removed | 44.3 | 66.6 | 64.8 | 50.5 |
| | `chapter rank` > 3 removed | 42.5 | 65.7 | 63.2 | 49.0 |
| | `sim` < 0.2 removed | 44.1 | 67.0 | 64.0 | 49.9 |

The filtered models fit their training data better (final training accuracy
48.9% to 51.7%, against 45.5%). They are worse on the classified data, which is
real parcel data. Top 5 falls by 2.0 to 3.6 points against the two baselines,
which differ from each other by 0.3 points. On good goods, the best filter
gains 1.3 to 2.8 points of top 1 over the baselines. The baseline seeds differ
by 1.5 points and the set is small (1 item is 0.25 points), so this gain is not
reliable.

## Conclusion

Do not filter on similarity to self-texts. The signal measures how much a
description reads like tariff text. It does not measure whether the code is
correct.

A noise filter must use an in-domain signal. These options are not tested:

- Out-of-fold disagreement (confident learning): train on folds, then flag
  rows where the held-out model gives high confidence to a different code.
- kNN label agreement: flag rows whose nearest training descriptions agree on
  a different chapter. The kNN index from experiment 001 supports this.
- An LLM or expert audit of a sample, to calibrate one of the above signals.

## Reproduce

Build the kNN index first (see experiment 001). Then run:

```sh
PYTHONPATH=. python experiments/semantic_filter_analysis.py
PYTHONPATH=. python experiments/train_filtered.py --rule chapter_rank_gt \
  --threshold 10 --output-dir benchmarking_data/experiments/002/chapter_rank_gt10
```

Copy the output `model.pt` to `target/` and run the usual benchmarks. The
analysis takes under a minute and each training run takes about 15 minutes on
an RTX 5090.
