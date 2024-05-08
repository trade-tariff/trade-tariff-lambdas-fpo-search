import logging
import os
from pathlib import Path
import pickle
import shutil
from typing import Optional, List, Tuple
import math
from sentence_transformers import SentenceTransformer
import torch
import fnv_c
from multiprocessing import Pool, cpu_count
from inference.infer import transformer

# Global variables for multiprocessing
global_model = None
global_torch_device = "cpu"
global_batch_size = 100


def init_worker(transformer_model: str, torch_device: str, batch_size: int):
    """Initialize the model in each worker process."""
    global global_model, global_torch_device, global_batch_size
    global_model = SentenceTransformer(transformer_model, device=torch_device)
    global_torch_device = torch_device
    global_batch_size = batch_size


def encode_texts(texts_batch: List[str]) -> List[torch.Tensor]:
    """Encodes a batch of texts using the global SentenceTransformer model."""
    return global_model.encode(
        texts_batch,
        batch_size=global_batch_size,
        show_progress_bar=False,
        convert_to_numpy=False,
        device=global_torch_device,
    )


def process_batch(args: Tuple[List[str], List[int]]) -> List[Tuple[int, torch.Tensor]]:
    """Helper function to process a batch concurrently."""
    batch_texts, batch_indexes = args
    batch_embeddings = encode_texts(batch_texts)
    return [
        (idx, embedding.cpu())
        for idx, embedding in zip(batch_indexes, batch_embeddings)
    ]


class EmbeddingsProcessor:
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        torch_device: str = "cpu",
        batch_size: int = 100,
        cache_checkpoint: int = 50000,
        transformer_model: str = transformer,
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
        self._transformer_model = transformer_model
        self._logger = logger

        self._load_cache()

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

    def create_embeddings(self, texts: List[str]) -> List[Optional[torch.Tensor]]:
        self._logger.info(f"ℹ️  Creating embeddings for {len(texts)} texts")
        sentence_embeddings = [None] * len(texts)

        texts_to_encode = []
        indexes_to_encode = []

        for idx, text in enumerate(texts):
            hash = fnv_c.fnv1a_64(str.encode(text))
            if self._cache is not None and hash in self._cache:
                sentence_embeddings[idx] = torch.Tensor(self._cache[hash])
            else:
                texts_to_encode.append(text)
                indexes_to_encode.append(idx)

        self._logger.info(f"ℹ️  Need to calculate {len(texts_to_encode)} uncached texts")

        max_checkpoint = math.ceil(len(texts_to_encode) / self._cache_checkpoint)

        # Prepare batches for parallel processing
        batches = [
            (
                texts_to_encode[
                    i * self._cache_checkpoint : (i + 1) * self._cache_checkpoint
                ],
                indexes_to_encode[
                    i * self._cache_checkpoint : (i + 1) * self._cache_checkpoint
                ],
            )
            for i in range(max_checkpoint)
        ]

        # Initialize multiprocessing pool with model and device info
        with Pool(
            processes=cpu_count(),
            initializer=init_worker,
            initargs=(self._transformer_model, self._torch_device, self._batch_size),
        ) as pool:
            results = pool.map(process_batch, batches)

        # Update sentence_embeddings and cache
        for batch_results in results:
            for idx, embedding in batch_results:
                sentence_embeddings[idx] = embedding

                if self._cache is not None:
                    self._cache[fnv_c.fnv1a_64(str.encode(texts[idx]))] = (
                        embedding.tolist()
                    )

        self._save_cache()

        return sentence_embeddings
