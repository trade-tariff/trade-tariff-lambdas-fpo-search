import logging
import unittest
from typing import List, Tuple

from training.cleaning_pipeline import Map2025CodesTo2026Codes

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = Map2025CodesTo2026Codes.build()


class TestMap2025To2026Codes(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        (
            ("2841908510", "10 digit changed"),
            ("28419040", "10 digit changed"),
        ),
        (
            ("29093038", "8 digit changed"),
            ("29093037", "8 digit changed"),
        ),
        (
            ("1234567890", "unchanged"),
            ("1234567890", "unchanged"),
        ),
    ]

    def test_filter(self):
        for example, expected in TestMap2025To2026Codes.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
