import logging
import unittest
from data_sources.static import StaticDataSource

from training.prepare_data import TrainingDataLoader

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

training_data_loader = TrainingDataLoader(logger=logger)


class Test_TrainingDataLoader_fetch_data(unittest.TestCase):
    def test_it_should_only_use_codes_from_code_creating_sources(self):
        code_data_source = StaticDataSource(
            [("Description 1", "1234567890"), ("Description 2", "1111111111")],
            creates_codes=True,
            authoritative=False,
        )

        non_code_data_source = StaticDataSource(
            [("Description 3", "1234567890"), ("Description 4", "2222222222")],
            creates_codes=False,
            authoritative=False,
        )

        (unique_texts, subheadings, texts, labels) = training_data_loader.fetch_data(
            [non_code_data_source, code_data_source]
        )

        self.assertEqual(
            unique_texts, ["description 3", "description 1", "description 2"]
        )
        self.assertEqual(subheadings, ["12345678", "11111111"])
        self.assertEqual(texts, [0, 1, 2])
        self.assertEqual(labels, [0, 0, 1])

    def test_codes_from_an_authoritative_source_should_override_other_codes(self):
        authoritative_data_source = StaticDataSource(
            [("Description 1", "1234567890"), ("Description 2", "1111111111")],
            creates_codes=True,
            authoritative=True,
        )

        non_authoritative_data_source = StaticDataSource(
            [("Description 2", "2222222222"), ("Description 3", "3333333333")],
            creates_codes=True,
            authoritative=False,
        )

        (unique_texts, subheadings, texts, labels) = training_data_loader.fetch_data(
            [non_authoritative_data_source, authoritative_data_source]
        )

        self.assertEqual(
            unique_texts, ["description 2", "description 3", "description 1"]
        )
        self.assertEqual(subheadings, ["22222222", "33333333", "12345678", "11111111"])
        self.assertEqual(texts, [0, 1, 2, 0])
        self.assertEqual(labels, [3, 1, 2, 3])
