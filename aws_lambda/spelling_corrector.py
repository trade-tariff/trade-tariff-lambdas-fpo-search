import os

from textblob.en import Spelling
from aws_lambda.quote_tokeniser import QuoteTokeniser


class SpellingCorrector:
    SPELLING_MODEL_FILEPATH = "config/data/spelling-model.txt"
    SPELLING_MODEL_FALLBACK_FILEPATH = "config/data/spelling-model-fallback.txt"

    def __init__(self):
        self._spelling = None
        self._maximum_word_length = int(os.getenv("MAXIMUM_WORD_LENGTH", "15"))

    def correct(self, term):
        if os.getenv("FPO_SPELLING_CORRECTION_ENABLED") == "False":
            return term
        else:
            self.load_spelling()

            word_tuples = QuoteTokeniser.tokenise(term)

            corrected_terms = []

            for word, quoted in word_tuples:
                if not quoted and len(word) > self._maximum_word_length:
                    return term

                if quoted:
                    corrected_terms.append(word)
                else:
                    corrected_term = self._spelling.suggest(word)[0][0]
                    corrected_terms.append(corrected_term)

            return " ".join(corrected_terms)

    def load_spelling(self):
        if not self._spelling:
            spelling = Spelling(SpellingCorrector.SPELLING_MODEL_FALLBACK_FILEPATH)

            self._spelling = spelling
