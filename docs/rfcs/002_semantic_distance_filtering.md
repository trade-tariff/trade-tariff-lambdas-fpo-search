# RFC-002: Data Quality Filtering Using Semantic Distance to Self-Texts

- RFC ID: RFC-002
- Title: Data Quality Filtering for Tradeset Descriptions Using Semantic Distance to Self-Texts
- Authors: [Will Fish]
- Status: Draft
- Created: November 27, 2025
- Updated: November 27, 2025
- Version: 1.0
- Target Component: Training data preparation pipeline (`training/cleaning_pipeline.py`)

## Abstract

This RFC proposes the introduction of a new, automated data cleaning step to improve the quality of the training data sourced from tradeset files (`raw_source_data/tradesets_descriptions/`). A significant portion (~50%) of these descriptions have been identified as semantically incorrect for their assigned commodity code. The proposed solution is to filter these low-quality examples by comparing their semantic meaning to the canonical "self-text" description for the same commodity code. This will be achieved by calculating the cosine similarity between the embeddings of the tradeset description and the corresponding self-text description, and discarding any examples where the similarity is below a configurable threshold.

## Motivation and Rationale

### Current Problem

The tradeset description files provide a large volume of valuable, real-world training data. However, manual sampling and analysis have revealed that a high percentage of these descriptions are "meaningfully wrong," where the text does not accurately describe the assigned commodity code.

While the current neural network model appears robust enough to "muddle through" this noise, relying on noisy data has several negative consequences:

- It likely degrades the model's accuracy and its ability to generalize to new, unseen descriptions.
- It may force the model to learn spurious or incorrect correlations.
- It makes model evaluation and debugging more difficult.

Cleaning this data manually or via large language model (LLM) calls is prohibitively expensive and time-consuming, given the volume of data. A programmatic, scalable, and cost-effective solution is required.

### Proposed Rationale

The official self-text descriptions from `reference_data/CN2025_SelfText_EN_DE_FR.csv` represent a trusted, human-curated "golden record" for what each commodity code represents. We can leverage the same sentence-transformer-based embeddings used in our model to measure how semantically "close" a tradeset description is to its assigned code's self-text.

By calculating the cosine similarity between the two description embeddings, we get a powerful, automated signal of data quality. A low similarity score indicates that the tradeset description has drifted significantly in meaning from the canonical description and is likely incorrect. This approach is highly relevant as it uses the same notion of "meaning" that the classification model itself relies on.

## Proposed Changes

We propose creating a new filter class to be added to the `training.cleaning_pipeline.CleaningPipeline`.

Add a new `SemanticSimilarityFilter` Class

- Load the `CN2025_SelfText_EN_DE_FR.csv` file.
- Use the project's configured `SentenceTransformer` to generate and cache embeddings for all English-language self-text descriptions.
- Store these embeddings in a dictionary for fast lookup: `self._self_text_embeddings = {commodity_code: embedding_vector}`.

For each row `(description, code)` passed to the filter:

- Generate an embedding for the input `description`.
- Look up the corresponding canonical embedding: `self_text_embedding = self._self_text_embeddings.get(code)`.
- **If a self-text embedding for that code exists:**
    - Calculate the cosine similarity between the `description` embedding and the `self_text_embedding`.
    - If the similarity score is **below** a configured threshold, the filter will reject the row (e.g., return `None`).
- **If no self-text exists or the score is above the threshold:**
    - The row passes the filter and is returned unmodified.

A new configuration parameter will be added to `search-config.toml` to control the filter's sensitivity (which I assume will need tuning as we discover the correct heuristic value):

- `minimum_semantic_similarity = 0.6` (value to be determined).

## Benefits of the Proposed Change

- Improved Data Quality: Systematically removes semantically incorrect training examples, leading to a more accurate and reliable model.
- Automated and Scalable: Provides a hands-off mechanism for cleaning large datasets.
- Cost-Effective: Leverages the existing embedding model and avoids expensive LLM calls.
- Tunable and Auditable: The filtering logic is controlled by a single, understandable threshold.

## Potential Risks and Considerations

- Over-filtering: A threshold that is too high could discard valid training examples that use synonyms or different phrasing from the official self-text. The heuristic development process is crucial to mitigate this.
- Self-Text Coverage: The filter can only be applied to commodity codes that have a corresponding self-text description. Data for codes not in the self-text file will pass through unfiltered.
- Speed: It may introduce some latency in the data processing pipeline due to embedding calculations, but this is expected to be manageable.

### Closing thoughts

We can compare the semantic similarity scores against the results of our recent gemini audit to see how well they correlate. This will help us choose an initial threshold and validate the approach before full implementation.

We should probably use this comparison as a benchmark as we tune the threshold over time, aiming to maximize the removal of "meaningfully wrong" examples while minimizing the loss of valid data.
