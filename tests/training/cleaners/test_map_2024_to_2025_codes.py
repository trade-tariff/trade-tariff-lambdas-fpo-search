import logging
from typing import List, Tuple
import unittest

from training.filters.map_2024_to_2025_codes import Map2024CodesTo2025Codes

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = Map2024CodesTo2025Codes()


class TestMap2024To2025Codes(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        (
            ("8521109510", "10 digit changed"),
            ("85211000", "10 digit changed"),
        ),
        (
            ("85211095", "8 digit changed"),
            ("85211000", "8 digit changed"),
        ),
        (
            ("1234567890", "unchanged"),
            ("1234567890", "unchanged"),
        ),
    ]

    def test_filter(self):
        for example, expected in TestMap2024To2025Codes.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
