import logging
import os
from pathlib import Path
import pickle
import shutil
import math
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch
import fnv_c


class EmbeddingsProcessor:
    def __init__(
        self,
        transformer_model: str,
        cache_path: Optional[Path] = None,
        torch_device: str = "cpu",
        batch_size: int = 100,
        cache_checkpoint: int = 50000,
        logger: logging.Logger = logging.getLogger(),
    ) -> None:
        if cache_path is None:
            self._cache_file = None
        else:
            self._cache_file = cache_path / f"embeddings-cache-{transformer_model}.pkl"
        self._cache = None
        self._torch_device = torch_device
        self._batch_size = batch_size
        self._cache_checkpoint = cache_checkpoint
        self._sentence_transformer_model = SentenceTransformer(transformer_model).to(torch_device)
        self._logger = logger

        self._load_cache()

    def create_embeddings(self, texts: list[str]):
        self._logger.info(f"ℹ️  Creating embeddings for {len(texts)} texts")
        # Initialize an empty list to store the embeddings
        sentence_embeddings = []

        texts_to_encode = []
        indexes_to_encode = []

        for idx, text in enumerate(texts):
            hash = fnv_c.fnv1a_64(str.encode(text))

            if self._cache is not None and hash in self._cache:
                sentence_embeddings.append(torch.Tensor(self._cache[hash]))
            else:
                sentence_embeddings.append(None)
                texts_to_encode.append(text)
                indexes_to_encode.append(idx)

        self._logger.info(f"ℹ️  Need to calculate {len(texts_to_encode)} uncached texts")

        max_checkpoint = int(math.ceil(len(texts_to_encode) / self._cache_checkpoint))

        # Process texts in batches
        for i in range(max_checkpoint):
            self._logger.info(
                f"ℹ️  Create embeddings - checkpoint {i + 1} of {max_checkpoint}..."
            )

            start_index = i * self._cache_checkpoint
            end_index = min((i + 1) * self._cache_checkpoint, len(texts_to_encode))

            batch_texts = texts_to_encode[start_index:end_index]
            batch_indexes = indexes_to_encode[start_index:end_index]

            # Encode the batch of texts
            batch_embeddings = self._sentence_transformer_model.encode(
                batch_texts,
                batch_size=self._batch_size,
                show_progress_bar=True,
                convert_to_numpy=False,
            )

            # Append the batch embeddings to the list
            for idx, embedding in enumerate(batch_embeddings):
                sentence_embeddings[batch_indexes[idx]] = embedding.cpu()

                if self._cache is not None:
                    self._cache[fnv_c.fnv1a_64(str.encode(batch_texts[idx]))] = (
                        embedding.tolist()
                    )

            self._save_cache()

            if self._torch_device == "mps":
                torch.mps.empty_cache()
            elif self._torch_device == "cuda":
                torch.cuda.empty_cache()

        return sentence_embeddings

    def _load_cache(self):
        if self._cache_file is not None:
            if os.path.isfile(self._cache_file):
                print("💾⇨ Loading embedding cache")
                with open(str(self._cache_file), "rb") as fp:
                    self._cache = pickle.load(fp)
            else:
                self._logger.info("ℹ️  Creating new embedding cache")
                self._cache = {}
        else:
            self._cache = None

    def _save_cache(self):
        if self._cache_file is not None:
            self._logger.info("💾⇦ Saving embedding cache")

            # Write to a temp file first so that we don't corrupt an existing file
            temp_file_path = str(self._cache_file) + ".tmp"

            with open(temp_file_path, "wb") as temp_file:
                pickle.dump(self._cache, temp_file)

            shutil.move(temp_file_path, self._cache_file)
