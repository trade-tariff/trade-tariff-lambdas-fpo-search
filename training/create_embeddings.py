import logging
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
        self._sentence_transformer_model = SentenceTransformer(transformer_model).to(
            torch_device
        )
        self._logger = logger

    def create_embeddings(self, texts: list[str]):
        self._sentence_transformer_model.to(self._torch_device)

        sentence_embeddings = self._sentence_transformer_model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,  # Avoid OOM issues for large datasets
        )

        sentence_embeddings = torch.from_numpy(sentence_embeddings)

        if self._torch_device == "mps":
            torch.mps.empty_cache()

        elif self._torch_device == "cuda":
            torch.cuda.empty_cache()

        return sentence_embeddings
