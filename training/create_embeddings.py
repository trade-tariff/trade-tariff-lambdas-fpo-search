import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import tqdm

sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def create_embeddings(texts: list[str], labels: list[int], torch_device: str = "cpu"):
    batch_size = 50

    embedding_chunks = []
    max_batch = np.ceil(len(texts) / batch_size)

    sentence_transformer_model.to(torch_device)

    sentence_embeddings = torch.empty(0, 384)

    # process each batch
    for cb in tqdm.tqdm(batched(texts, batch_size), total=max_batch):
        sentence_embeddings = torch.cat((sentence_embeddings, sentence_transformer_model.encode(cb, convert_to_tensor=True, batch_size=batch_size, device=torch_device).to('cpu'))) # type: ignore

    return sentence_embeddings