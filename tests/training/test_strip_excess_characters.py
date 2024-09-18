import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import StripExcessCharacters

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = StripExcessCharacters()


class TestStripExcessCharacters(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        # strips additional whitespace between words
        (
            ("99999999", " any    description  at all "),
            ("99999999", "any description at all"),
        ),
        # strips whitespace
        (
            (" 99999999 ", " any description at all "),
            ("99999999", "any description at all"),
        ),
        # rstrips newlines
        (
            (" 99999999\n", " any description at all\n "),
            ("99999999", "any description at all"),
        ),
        # rstrips tabs
        (
            (" 99999999\t", " any description at all\t "),
            ("99999999", "any description at all"),
        ),
        # rstrips carriage returns
        (
            (" 99999999\r", " any description at all\r "),
            ("99999999", "any description at all"),
        ),
        # rstrips periods
        (
            (" 99999999.", " any description at all."),
            ("99999999", "any description at all"),
        ),
        # rstrips commas
        (
            (" 99999999,", " any description at all,"),
            ("99999999", "any description at all"),
        ),
    ]

    def test_filter(self):
        for example, expected in TestStripExcessCharacters.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
