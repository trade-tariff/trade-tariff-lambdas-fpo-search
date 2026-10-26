import torch
from torch import Tensor


def build_knn_index(
    unique_text_values: list[str],
    unique_embeddings: Tensor,
    text_indexes: list[int],
    labels: list[int],
    num_labels: int,
) -> dict:
    """Store each unique description's embedding with its label counts.

    The counts include each data source's multiplier, so they carry the same
    weighting the network sees in training. Label counts are stored in CSR form,
    sorted by text index.
    """
    texts = torch.tensor(text_indexes, dtype=torch.int64)
    label_tensor = torch.tensor(labels, dtype=torch.int64)

    keys, counts = torch.unique(texts * num_labels + label_tensor, return_counts=True)
    pair_texts = keys // num_labels
    pair_labels = keys % num_labels

    offsets = torch.zeros(len(unique_text_values) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(
        torch.bincount(pair_texts, minlength=len(unique_text_values)), dim=0
    )

    return {
        "texts": unique_text_values,
        "embeddings": unique_embeddings.to(torch.float16),
        "offsets": offsets,
        "labels": pair_labels.to(torch.int32),
        "counts": counts.to(torch.int32),
    }
