from logging import Logger
import logging
from math import floor
from aws_lambda_powertools import Logger as AWSLogger

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from model.model import SimpleNN

from train_args import TrainScriptArgsParser

args = TrainScriptArgsParser()
args.load_config_file()

score_cutoff = 0.01 # We won't send back any results with a score lower than this
top_n_softmax_percent = 0.05  # We only softmax over the top 5% of results to ignore the long tail of nonsense ones
cumulative_cutoff = 0.9
vague_term_code = "vvvvvvvvvv"


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
        offline: bool = False,
        logger: Logger | AWSLogger = logging.getLogger("inference"),
    ) -> None:
        super().__init__()

        self._subheadings = subheadings
        self._device = device
        self._logger = logger

        # Load the model from disk
        self._model = self.load_model().to(self._device)

        logger.info(
            f"💾⇨ Sentence Transformers running in {'Offline' if offline else 'Online'} mode"
        )

        self._sentence_transformer_model = SentenceTransformer(
            args.transformer(), device=self._device, local_files_only=offline
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
            normalize_embeddings=True,
        )

        # Run it through the model to get the predictions
        results = self._model(new_embeddings)

        predictions_to_digits = {}

        for i, prediction in enumerate(results[0]):
            code = str(self._subheadings[i])[:digits]
            score = prediction.item()

            if code in predictions_to_digits:
                predictions_to_digits[code] += score
            else:
                predictions_to_digits[code] = score

        top_results = sorted(
            predictions_to_digits.items(), key=lambda x: x[1], reverse=True
        )

        top_results = top_results[: floor(len(top_results) * top_n_softmax_percent)]

        # Extract values
        values = np.array([val for _, val in top_results])

        # Compute softmax
        exp_values = np.exp(
            values - np.max(values)
        )  # Subtract max for numerical stability
        softmax_values = exp_values / np.sum(exp_values)

        max = np.max(softmax_values)
        min_confidence = 0.5

        softmax_results = [
            (category, softmax)
            for (category, _), softmax in zip(top_results, softmax_values)
            if max * min_confidence <= softmax
        ]

        result = []

        vague_term_truncated = vague_term_code[:digits]

        cumulative_score = 0

        for i in softmax_results:
            classification = ClassificationResult(i[0], i[1])
            # If the score is less than the cutoff then stop iterating through
            if classification.score < score_cutoff:
                break

            # If we've hit the vague terms code then we'll skip it
            if classification.code == vague_term_truncated:
                continue

            result.append(classification)

            cumulative_score += classification.score

            if len(result) >= limit:
                break

            if cumulative_score >= cumulative_cutoff:
                break

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
            model.load_state_dict(torch.load(model_file, map_location=self._device))
        except Exception as e:
            self._logger.error(f"Failed to load the model: {e}")
            raise e

        model.eval()

        self._logger.info("🧠⚡ Model loaded")

        return model
