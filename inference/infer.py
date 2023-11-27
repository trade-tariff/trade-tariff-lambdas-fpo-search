import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch


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
        self, model_file: Path, subheadings: list[str], device: str = "cpu"
    ) -> None:
        super().__init__()

        transformer = os.environ.get(
            "SENTENCE_TRANSFORMER_PRETRAINED_MODEL", "all-MiniLM-L6-v2"
        )
        transformer_cache_directory = os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME", "/tmp/sentence_transformers/"
        )
        sentence_transformer_model_directory = (
            transformer_cache_directory + "sentence-transformers_" + transformer
        )
        print(
            f"💾⇨ Sentence Transformer cache directory: {sentence_transformer_model_directory}"
        )

        self._subheadings = subheadings
        self._device = device

        # Load the model from disk
        print(f"💾⇨ Loading model file: {model_file}")
        self._model = torch.load(model_file)
        print("Model loaded")

        # Use predownloaded transformer if available
        if sentence_transformer_model_directory:
            print(
                f"💾⇨ Loading Sentence Transformer model from {sentence_transformer_model_directory}"
            )

            exists = os.path.isdir(sentence_transformer_model_directory)
            print(f"💾⇨ Sentence Transformer model exists: {exists}")
            self._sentence_transformer_model = SentenceTransformer(
                sentence_transformer_model_directory,
            )
        else:
            print(f"💾⇨ Downloading Sentence Transformer model {transformer}")
            # Otherwise download it from the HuggingFace model hub
            self._sentence_transformer_model = SentenceTransformer(transformer)

    def classify(
        self, search_text: str, limit: int = 5, digits: int = 6
    ) -> list[ClassificationResult]:
        # Make sure the model is on the correct device
        self._model.to(self._device)

        # Fetch the embedding for the search text
        new_texts = [search_text]
        new_embeddings = self._sentence_transformer_model.encode(
            new_texts, convert_to_tensor=True
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
            result.append(ClassificationResult(i[0], i[1]))

        return result
