import logging
from logging import Logger
from math import floor

import numpy as np
import toml
import torch
from aws_lambda_powertools import Logger as AWSLogger
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer

from model.model import SimpleNN
from quantize_model import load_model, quantize_model
from train_args import TrainScriptArgsParser

args = TrainScriptArgsParser()
args.load_config_file()

score_cutoff = 0.01  # We won't send back any results with a score lower than this
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
        self,
        search_text: str,
        limit: int = 5,
        digits: int = 6,
    ) -> list[ClassificationResult]:
        raise NotImplementedError()


class FlatClassifier(Classifier):
    def __init__(
        self,
        subheadings: list[str],
        device: str,
        offline: bool = False,
        model: SimpleNN | None = None,
        logger: Logger | AWSLogger = logging.getLogger("inference"),
    ) -> None:
        super().__init__()

        self._subheadings = subheadings
        self._device = device
        self._logger = logger

        # Load the model from disk
        self._model = self.load_model().to(self._device) if model is None else model

        logger.info(
            f"💾⇨ Sentence Transformers running in {'Offline' if offline else 'Online'} mode"
        )

        self._sentence_transformer_model = SentenceTransformer(
            args.transformer(), device=self._device, local_files_only=offline
        )

    def classify(
        self,
        search_text: str,
        limit: int = 5,
        digits: int = 6,
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

        # Run it through the model to get the predictions (with no_grad for inference efficiency)
        with torch.no_grad():
            results = self._model(new_embeddings).detach().numpy()

        logits = results[0]  # 1D NumPy array of logits for the single input

        groups = {}

        for i, logit in enumerate(logits):
            code = str(self._subheadings[i])[:digits]
            if code not in groups:
                groups[code] = []
            groups[code].append(logit)

        predictions_to_digits = {
            code: logsumexp(group_logits) for code, group_logits in groups.items()
        }
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

        max_val = np.max(softmax_values)
        min_confidence = 0.05

        # Cutoff everything below confidence level of the top result
        softmax_results = [
            (category, softmax)
            for (category, _), softmax in zip(top_results, softmax_values)
            if max_val * min_confidence <= softmax
        ]

        result = []

        vague_term_truncated = vague_term_code[:digits]

        cumulative_score = 0

        for i in softmax_results:
            classification = ClassificationResult(i[0], float(i[1]))
            # If the score is less than the cutoff then stop iterating through
            if classification.score < score_cutoff:
                break

            # If we've hit the vague terms code then we'll skip it
            if classification.code == vague_term_truncated:
                break

            result.append(classification)

            cumulative_score += classification.score

            if len(result) >= limit:
                break

            if cumulative_score >= cumulative_cutoff:
                break

        return result

    def load_model(self):
        model_file = args.target_dir() / "model.pt"
        model_qauntized_file = args.target_dir() / "model_quantized.pt"
        model_config_file = args.target_dir() / "model.toml"

        if args.uses_quantized_model():
            self._logger.info(f"💾⇨ Loading model file: {model_qauntized_file}")
            if not model_qauntized_file.exists():
                model = load_model()
                quantize_model(model)

            torch.serialization.add_safe_globals([SimpleNN])
            try:
                model = torch.load(
                    model_qauntized_file,
                    map_location=self._device,
                    weights_only=False,
                )
            except Exception as e:
                self._logger.error(f"Failed to load the model: {e}")
                raise e
        else:
            self._logger.info(f"💾⇨ Loading model file: {model_file}")
            model_config = toml.load(model_config_file)

            model = SimpleNN(
                model_config["input_size"],
                model_config["hidden_size"],
                model_config["output_size"],
                model_config["dropout_layer_1_percentage"],
                model_config["dropout_layer_2_percentage"],
            )

            model.load_state_dict(torch.load(model_file, map_location=self._device))

        model.eval()
        self._logger.info("🧠⚡ Model loaded")

        return model
