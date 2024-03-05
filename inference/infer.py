from logging import Logger
import logging
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

score_cutoff = 0.05  # We won't send back any results with a score lower than this


class ClassificationResult:
    def __init__(self, code: str, score: float) -> None:
        self.code = code
        self.score = score

    def __repr__(self) -> str:
        return "%s = %.2f" % (self.code, self.score * 1000)


class Classifier:
    def classify(
        self, search_text: str, limit: int = 5, digits: int = 6
    ) -> list[ClassificationResult]:
        raise NotImplementedError()


class FlatClassifier(Classifier):
    def __init__(
        self,
        model_file: Path,
        subheadings: list[str],
        device: torch.device,
        logger: Logger = logging.getLogger(),
    ) -> None:
        super().__init__()

        transformer = os.environ.get(
            "SENTENCE_TRANSFORMER_PRETRAINED_MODEL", "all-mpnet-base-v2"
        )
        transformer_cache_directory = os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME", "/tmp/sentence_transformers/"
        )
        sentence_transformer_model_directory = (
            transformer_cache_directory + "sentence-transformers_" + transformer
        )
        logger.info(
            f"💾⇨ Sentence Transformer cache directory: {sentence_transformer_model_directory}"
        )

        self._subheadings = subheadings
        self._device = device
        self._logger = logger

        # Load the model from disk
        logger.info(f"💾⇨ Loading model file: {model_file}")

        # Make sure the model is on the correct device
        self._model = torch.load(model_file)
        logger.info("🧠⚡ Model loaded")

        # Use predownloaded transformer if available
        if Path(sentence_transformer_model_directory).exists():
            logger.info(
                f"💾⇨ Loading Sentence Transformer model from {sentence_transformer_model_directory}"
            )

            exists = os.path.isdir(sentence_transformer_model_directory)
            logger.info(f"💾⇨ Sentence Transformer model exists: {exists}")
            self._sentence_transformer_model = SentenceTransformer(
                sentence_transformer_model_directory, device=device
            )
        else:
            logger.info(f"💾⇨ Downloading Sentence Transformer model {transformer}")
            # Otherwise download it from the HuggingFace model hub
            self._sentence_transformer_model = SentenceTransformer(
                transformer, device=device
            )

    def classify(
        self, search_text: str, limit: int = 5, digits: int = 6
    ) -> list[ClassificationResult]:
        # Fetch the embedding for the search text
        new_texts = [search_text]
        new_embeddings = self._sentence_transformer_model.encode(
            new_texts,
            convert_to_tensor=True,
            device=self._device,
            show_progress_bar=False,
        )

        # Run it through the model to get the predictions
        predictions = torch.nn.functional.softmax(self._model(new_embeddings), dim=1)

        predictions_to_digits = {}

        for i, prediction in enumerate(predictions[0]):
            code = str(self._subheadings[i])[:digits]
            score = prediction.item()

            if code in predictions_to_digits:
                predictions_to_digits[code] += score
            else:
                predictions_to_digits[code] = score

        top_results = sorted(
            predictions_to_digits.items(), key=lambda x: x[1], reverse=True
        )[:limit]

        result = []

        for i in top_results:
            # If the score is less than the cutoff then stop iterating through
            if i[1] < score_cutoff:
                break

            result.append(ClassificationResult(i[0], i[1]))

        return result
