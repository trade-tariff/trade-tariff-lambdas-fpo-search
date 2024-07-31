from logging import Logger
import logging
from pathlib import Path
import time

import torch
from sentence_transformers import SentenceTransformer
from model.model import SimpleNN

from train_args import TrainScriptArgsParser


args = TrainScriptArgsParser()
args.load_config_file()

score_cutoff = 0.05  # We won't send back any results with a score lower than this
vague_term_code = "vvvvvvvvvv"


class ClassificationResult:
    def __init__(self, code: str, score: float) -> None:
        self.code = code
        self.score = score

    def __repr__(self) -> str:
        return "%s = %.2f" % (self.code, self.score * 1000)


class Classifier:
    def classify(self, search_text: str, limit: int = 5, digits: int = 6) -> list[ClassificationResult]:
        raise NotImplementedError()


class FlatClassifier(Classifier):
    def __init__(self, subheadings: list[str], device: str, logger: Logger = logging.getLogger("inference")) -> None:
        super().__init__()

        self._subheadings = subheadings
        self._device = device
        self._logger = logger

        # Load the model from disk
        self._model = self.load_model().to(self._device)
        self._sentence_transformer_model = self.load_sentence_transformer()

        logger.info(f"💾⇨ Sentence Transformer cache directory: {args.transformer_cache_directory()}")

    def classify(self, search_text: str, limit: int = 5, digits: int = 6) -> list[ClassificationResult]:
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

        top_results = sorted(predictions_to_digits.items(), key=lambda x: x[1], reverse=True)[:limit]

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

    def load_model(self):
        model_file = args.target_dir() / "model.pt"

        self._logger.info(f"💾⇨ Loading model file: {model_file}")

        model = SimpleNN(
            args.model_input_size(),
            args.model_hidden_size(),
            args.model_output_size(),
            args.model_dropout_layer_1_percentage(),
            args.model_dropout_layer_2_percentage(),
        )

        try:
            t = time.perf_counter()
            model.load_state_dict(torch.load(model_file, map_location=self._device))
            self._logger.info(f"💾⇨ Classification model loaded in {time.perf_counter() - t} seconds")
        except Exception as e:
            self._logger.error(f"Failed to load the model: {e}")
            raise e

        model.eval()

        self._logger.info("🧠⚡ Model loaded")

        return model

    def load_sentence_transformer(self) -> torch.nn.Sequential:
        model_file = Path(args.transformer_cache_directory()) / f"{args.transformer()}_transformer_model.pt"

        self._logger.info(f"Checking if cached model file exists: {str(model_file)} ... {model_file.is_file()}")

        if model_file.is_file():
            self._logger.info(f"💾⇨ Loading sentence transformer cached model from {str(model_file)}")
            t = time.perf_counter()

            model = torch.load(model_file, map_location=self._device)
            self._logger.info(f"💾⇨ Sentence transformer cached model loaded in {time.perf_counter() - t} seconds")

            return model
        else:
            self._logger.info(f"💾⇨ Downloading sentence transformer model {args.transformer()}")

            # Otherwise download it from the HuggingFace model hub
            t = time.perf_counter()
            model = SentenceTransformer(args.transformer(), device=self._device)
            self._logger.info(f"🛜⇨ Sentence transformer (down)loaded in {time.perf_counter() - t} seconds")

            return model
