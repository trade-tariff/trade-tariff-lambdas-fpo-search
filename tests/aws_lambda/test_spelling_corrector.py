import logging
import unittest
import os
from aws_lambda.spelling_corrector import SpellingCorrector
from unittest import mock

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Test_spelling_corrector(unittest.TestCase):
    def test_spelling_corrector_correct(self):
        spelling_corrector = SpellingCorrector()
        search_query = (
            "halbiut sausadge stenolepsis chese bnoculars parnsip farmacy papre"
        )
        corrected_search_query = spelling_corrector.correct(search_query)

        self.assertEqual(
            corrected_search_query,
            "halibut sausage stenolepis cheese binoculars parsnip pharmacy paper",
        )

    def test_spelling_corrector_with_single_quotes(self):
        spelling_corrector = SpellingCorrector()
        search_query = (
            "'halbiut' sausadge stenolepsis chese bnoculars parnsip farmacy papre"
        )
        corrected_search_query = spelling_corrector.correct(search_query)

        self.assertEqual(
            corrected_search_query,
            "'halbiut' sausage stenolepis cheese binoculars parsnip pharmacy paper",
        )

    def test_spelling_corrector_with_double_quotes(self):
        spelling_corrector = SpellingCorrector()
        search_query = (
            '"halbiut" sausadge stenolepsis chese bnoculars parnsip farmacy papre'
        )
        corrected_search_query = spelling_corrector.correct(search_query)

        self.assertEqual(
            corrected_search_query,
            '"halbiut" sausage stenolepis cheese binoculars parsnip pharmacy paper',
        )

    def test_spelling_corrector_synonym_not_corrected(self):
        spelling_corrector = SpellingCorrector()
        search_query = "acamol"
        corrected_search_query = spelling_corrector.correct(search_query)

        self.assertEqual(corrected_search_query, "acamol")

    @mock.patch.dict(
        os.environ, {"FPO_SPELLING_CORRECTION_ENABLED": "False"}, clear=True
    )
    def test_spelling_corrector_env_variable(self):
        spelling_corrector = SpellingCorrector()

        search_query = "farmacy"
        corrected_search_query = spelling_corrector.correct(search_query)

        self.assertEqual(corrected_search_query, "farmacy")
