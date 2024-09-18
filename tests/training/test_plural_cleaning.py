import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import PluralCleaning

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = PluralCleaning()


class TestPluralCleaning(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        # preserves size s
        (
            ("99999999", "size s"),
            ("99999999", "size s"),
        ),
        # removes spacing between word and s
        (
            ("99999999", "women s"),
            ("99999999", "womens"),
        ),
    ]

    def test_filter(self):
        for example, expected in TestPluralCleaning.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
