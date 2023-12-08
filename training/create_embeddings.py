import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import tqdm

sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")


def batched(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def create_embeddings(texts: list[str], torch_device: str = "cpu"):
    batch_size = 50

    max_batch = np.ceil(len(texts) / batch_size)

    sentence_transformer_model.to(torch_device)

    sentence_embeddings = torch.empty(0, 384)

    # process each batch
    for cb in tqdm.tqdm(batched(texts, batch_size), total=max_batch):
        sentence_embeddings = torch.cat(
            (
                sentence_embeddings,
                sentence_transformer_model.encode(
                    cb,
                    convert_to_tensor=True,
                    batch_size=batch_size,
                    device=torch_device,
                    show_progress_bar=False,
                ).to("cpu"),  # type: ignore
            )
        )  # type: ignore

    return sentence_embeddings
