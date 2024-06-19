import logging
import math
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingsProcessor:
    def __init__(
        self,
        transformer_model: str,
        torch_device: str = "cpu",
        batch_size: int = 100,
        logger: logging.Logger = logging.getLogger("embeddings"),
    ) -> None:
        self._torch_device = torch_device
        self._batch_size = batch_size
        self._sentence_transformer_model = SentenceTransformer(transformer_model).to(torch_device)
        self._logger = logger

    def create_embeddings(self, texts: list[str]):
        max_batch = int(math.ceil(len(texts) / self._batch_size))

        self._sentence_transformer_model.to(self._torch_device)

        # Initialize an empty list to store the embeddings
        sentence_embeddings = []

        # Process texts in batches
        for i in range(max_batch):
            start_index = i * self._batch_size
            end_index = min((i + 1) * self._batch_size, len(texts))

            batch_texts = texts[start_index:end_index]

            # Encode the batch of texts
            batch_embeddings = self._sentence_transformer_model.encode(
                batch_texts,
                batch_size=self._batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
            )


            # Append the batch embeddings to the list
            sentence_embeddings.extend(batch_embeddings)

            if self._torch_device == "mps":
                torch.mps.empty_cache()
            elif self._torch_device == "cuda":
                torch.cuda.empty_cache()

        return sentence_embeddings
