import unittest
import logging
import csv

from data_sources.commodities import CommoditiesDataSource
from training.cleaning_pipeline import (
    CleaningPipeline,
    NegationCleaning,
    RemoveEmptyDescription,
    RemoveShortDescription,
    RemoveSubheadingsNotMatchingRegexes,
    StripExcessCharacters,
)

filters = [
    StripExcessCharacters(),
    RemoveEmptyDescription(),
    RemoveShortDescription(min_length=4),
    RemoveSubheadingsNotMatchingRegexes(regexes=[r"^\d{8}$"]),
    NegationCleaning.build(),
]
pipeline = CleaningPipeline(filters)

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


class Test_CommoditiesDataSource(unittest.TestCase):
    with open("tests/data_sources/commodities.csv", "r") as f:
        content = f.read().splitlines()
        reader = csv.reader(content)

    def test_get_codes(self):
        data_source = CommoditiesDataSource(service="uk")
        data_source._reader = lambda: self.reader

        self.assertEqual(
            data_source.get_codes(8),
            {
                "01012100": {
                    "LIVE ANIMALS Live horses, asses, mules and hinnies Horses Pure-bred breeding animals"
                },
                "01012910": {
                    "LIVE ANIMALS Live horses, asses, mules and hinnies Horses Other For slaughter"
                },
            },
        )

    def test_get_codes_with_cleaning_pipeline(self):
        data_source = CommoditiesDataSource(service="uk", cleaning_pipeline=pipeline)
        data_source._reader = lambda: self.reader

        self.assertEqual(
            data_source.get_codes(8),
            {},
        )
