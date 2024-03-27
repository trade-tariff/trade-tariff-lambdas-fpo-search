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

    def test_that_earlier_sources_override_later_ones(self):
        first_data_source = StaticDataSource(
            [("Description 1", "1234567890"), ("Description 2", "1111111111")],
            creates_codes=True,
            authoritative=True,
        )

        second_data_source = StaticDataSource(
            [("Description 2", "2222222222"), ("Description 3", "3333333333")],
            creates_codes=True,
            authoritative=True,
        )

        (unique_texts, subheadings, texts, labels) = training_data_loader.fetch_data(
            [first_data_source, second_data_source]
        )

        self.assertEqual(
            unique_texts, ["description 1", "description 2", "description 3"]
        )
        self.assertEqual(subheadings, ["12345678", "11111111", "22222222", "33333333"])
        self.assertEqual(texts, [0, 1, 1, 2])
        self.assertEqual(labels, [0, 1, 1, 3])

    def test_that_it_ignores_data_with_too_few_digits(self):
        first_data_source = StaticDataSource(
            [("Description 1", "1111"), ("Description 2", "2222222222")],
            creates_codes=True,
            authoritative=False,
        )

        second_data_source = StaticDataSource(
            [("Description 3", "33333333"), ("Description 4", "444444")],
            creates_codes=True,
            authoritative=False,
        )

        (unique_texts, subheadings, texts, labels) = training_data_loader.fetch_data(
            [first_data_source, second_data_source], digits=6
        )

        self.assertEqual(
            unique_texts, ["description 2", "description 3", "description 4"]
        )
        self.assertEqual(subheadings, ["222222", "333333", "444444"])
        self.assertEqual(texts, [0, 1, 2])
        self.assertEqual(labels, [0, 1, 2])

    def test_that_it_ignores_blank_descriptions(self):
        data_source = StaticDataSource(
            [
                ("Description 1", "1111111111"),
                ("", "2222222222"),
                ("      ", "3333333333"),
            ],
            creates_codes=True,
            authoritative=False,
        )

        (unique_texts, subheadings, texts, labels) = training_data_loader.fetch_data(
            [data_source], digits=8
        )

        self.assertEqual(unique_texts, ["description 1"])
        self.assertEqual(subheadings, ["11111111"])
        self.assertEqual(texts, [0])
        self.assertEqual(labels, [0])

    def test_that_multiplier_works(self):
        data_source = StaticDataSource(
            [
                ("Description 1", "1111111111"),
            ],
            creates_codes=True,
            authoritative=False,
            multiplier=2,
        )

        (unique_texts, subheadings, texts, labels) = training_data_loader.fetch_data(
            [data_source], digits=8
        )

        self.assertEqual(unique_texts, ["description 1"])
        self.assertEqual(subheadings, ["11111111"])
        self.assertEqual(texts, [0, 0])
        self.assertEqual(labels, [0, 0])

    def test_that_multiplier_works_for_overridden_codes(self):
        authoritative_data_source = StaticDataSource(
            [("Description 1", "1234567890")],
            creates_codes=True,
            authoritative=True,
        )

        non_authoritative_data_source = StaticDataSource(
            [("Description 2", "2222222222"), ("Description 1", "3333333333")],
            creates_codes=True,
            authoritative=False,
            multiplier=2,
        )

        (unique_texts, subheadings, texts, labels) = training_data_loader.fetch_data(
            [authoritative_data_source, non_authoritative_data_source], digits=8
        )

        self.assertEqual(unique_texts, ["description 1", "description 2"])
        self.assertEqual(subheadings, ["12345678", "22222222", "33333333"])
        self.assertEqual(texts, [0, 1, 1, 0, 0])
        self.assertEqual(labels, [0, 1, 1, 0, 0])
