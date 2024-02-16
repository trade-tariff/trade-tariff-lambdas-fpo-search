import numpy as np
from sentence_transformers import SentenceTransformer
import torch

sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")


def batched(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def create_embeddings(texts: list[str], torch_device: str = "cpu"):
    # Define batch size
    batch_size = 2064

    max_batch = int(np.ceil(len(texts) / batch_size))

    sentence_transformer_model.to(torch_device)

    # Initialize an empty list to store the embeddings
    sentence_embeddings = []

    # Process texts in batches
    for i in range(max_batch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(texts))

        batch_texts = texts[start_index:end_index]

        # Encode the batch of texts
        batch_embeddings = sentence_transformer_model.encode(
            batch_texts, show_progress_bar=True
        )

        # Append the batch embeddings to the list
        sentence_embeddings.extend(batch_embeddings)

    sentence_embeddings = torch.tensor(sentence_embeddings)

    return sentence_embeddings
