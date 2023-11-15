from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class ClassificationResult:
    def __init__(self, code: str, score: float) -> None:
        self.code = code
        self.score = score

    def __repr__(self) -> str:
        return "%s = %.2f%%" % (self.code, self.score * 100)


class Classifier:
    def classify(self, search_text: str, limit: int = 5) -> list[ClassificationResult]:
        raise NotImplementedError()


class FlatClassifier(Classifier):
    def __init__(
        self, model_file: Path, subheadings: list[str], device: str = "cpu"
    ) -> None:
        super().__init__()

        self._subheadings = subheadings
        self._device = device

        # Load the model from disk
        print(f"💾⇨ Loading model file: {model_file}")
        self._model = torch.load(model_file)
        print("Model loaded")

        self._sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    def classify(self, search_text: str, limit: int = 5) -> list[ClassificationResult]:
        # Make sure the model is on the correct device
        self._model.to(self._device)

        # Fetch the embedding for the search text
        new_texts = [search_text]
        new_embeddings = self._sentence_transformer_model.encode(
            new_texts, convert_to_tensor=True
        )

        # Run it through the model to get the predictions
        predictions = torch.nn.functional.softmax(self._model(new_embeddings), dim=1)

        np_predictions = predictions.detach().numpy()

        top_results = np.argsort(np_predictions, axis=1)[:, -limit:]

        result = []

        for i in reversed(top_results[0]):
            result.append(
                ClassificationResult(
                    str(self._subheadings[i]), predictions[0][i].item()
                )
            )

        return result
