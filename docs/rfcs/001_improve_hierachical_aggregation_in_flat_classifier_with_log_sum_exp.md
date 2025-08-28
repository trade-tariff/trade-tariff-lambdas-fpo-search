# Improve Hierarchical Aggregation in FlatClassifier Using LogSumExp

- RFC ID: [RFC-001]
- Title: Switch to LogSumExp for Score Aggregation in FlatClassifier's Classify Method
- Authors: [Will Fish]
- Status: Draft
- Created: August 26, 2025
- Updated: August 26, 2025
- Version: 1.0
- Target Component: FlatClassifier in the classification pipeline

## Abstract

We currently aggregate scores (logits) output by the flat classifier. These scores get accumulated under the requested number of digits (usually 6). The accumulation mechanism currently is to sum the scores. This is problematic because we reinforce scores for specific references which leads to negative scores for other nodes in the same semantic area (siblings). This RFC proposes replacing the current summation of raw logits in the `FlatClassifier.classify` method with a logsumexp operation. This change addresses issues where strongly reinforced matches at full precision (8 digits) fail to propagate to higher-level aggregations due to negative contributions from sibling subheadings. The new approach preserves probabilistic intent, improves accuracy for reinforced training examples, and maintains numerical stability without introducing new dependencies.

## Motivation and Rationale

### Current Behavior and Problems

The existing implementation in `FlatClassifier.classify` accumulates scores by summing raw logits (model outputs) for all subheadings sharing a common prefix when truncating to fewer digits (e.g., 6 digits). For example:

- Raw logits are generated for each 8-digit subheading
- For a 6-digit code like '620462' (trousers), the score is the sum of logits from all matching 8-digit children (e.g. '62046239', '62046211', etc)

This leads to unintended dilution of strong signals:

- Logits for non-matching classes are often negative to suppress incorrect predictions during softmax
- Reinforced training (e.g., repeating examples 5 times for `extra_references.csv`) boosts the logit for the exact match (e.g., 8.3 for '62046239' on input "trousers") but amplifies negatives for siblings under the same prefix.
- Result: The aggregated sum for the parent code can become low or negative (e.g., 8.3 + multiple -1.5 values = negative), causing it to rank below unrelated groups with milder positives. This explains why '620462' disappears from top results at 6 digits despite dominating at 8 digits.

Summing raw logits is mathematically inappropriate for hierarchical aggregation:

- It treats negatives as penalties that cancel positives, rather than ignoring them (as they represent low-probability alternatives)
- This mismatches the probabilistic nature of softmax-based classification, where group likelihood should approximate the union of child probabilities.

## Benefits of the Proposed Change

- Preserves Strong Signals - Logsumexp computes log(sum(exp(logits))), effectively approximating the maximum logit in a group (for disparate values)
- Improved Accuracy - Aligns with hierarchical intent, ensuring reinforced matches bubble up. For "trousers", '620462' should now appear in top aggregated results.
- No New Dependencies
- Computationally similar to summation (e.g. no expected slow down)
- Backward Compatibility

## Evidence

- Fixes the specific set of issues
- Intuitively makes more sense
- Aligns with standard practices in ML

## Proposed Changes

- In `classify`, after obtaining raw logits as a NumPy array:
  1. Group logits by truncated code prefix.
  2. For each group, compute logsumexp instead of sum.
  3. Proceed with sorting, top-N selection, softmax, and filtering as before (now on these "group logits").

See also:

- https://lorenlugosch.github.io/posts/2020/06/logsumexp/
- https://en.wikipedia.org/wiki/LogSumExp
