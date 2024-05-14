from logging import Logger
import logging
import os
from pathlib import Path
import toml

import torch
from sentence_transformers import SentenceTransformer
from model.model import SimpleNN

score_cutoff = 0.05  # We won't send back any results with a score lower than this
vague_term_code = "vvvvvvvvvv"

transformer = os.environ.get(
    "SENTENCE_TRANSFORMER_PRETRAINED_MODEL", "all-MiniLM-L6-v2"
)
transformer_cache_directory = os.environ.get(
    "SENTENCE_TRANSFORMERS_HOME", "/tmp/sentence_transformers/"
)
sentence_transformer_model_directory = (
    transformer_cache_directory + "sentence-transformers_" + transformer
)

logger: Logger = (logging.getLogger(),)


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
        subheadings: list[str],
        device: str,
    ) -> None:
        super().__init__()

        logger.info(
            f"💾⇨ Sentence Transformer cache directory: {sentence_transformer_model_directory}"
        )

        self._subheadings = subheadings
        self._device = torch.device(device)
        self._logger = logger

        # Load the model from disk
        self._model = self.load_current_model().to(self._device)

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

            # If we've hit the vague terms code then we'll stop iterating through
            if i[0] == vague_term_code[:digits]:
                break

            result.append(ClassificationResult(i[0], i[1]))

        return result

    def load_current_model(self):
        if self._model is not None:
            return self._model

        model_config = toml.load("search_config.toml")
        model_file = model_config["model_file"]
        model_input_size = model_config["model_input_size"]
        model_hidden_size = model_config["model_hidden_size"]
        model_output_size = model_config["model_output_size"]

        logger.info(f"💾⇨ Loading model file: {model_file}")

        model = SimpleNN(model_input_size, model_hidden_size, model_output_size)
        try:
            model.load_state_dict(torch.load(model_file, map_location=self._device))
        except Exception as e:
            logger.error(f"Failed to load the model: {e}")
            raise e
        model.eval()

        self._model = model
        logger.info("🧠⚡ Model loaded")

        return model
