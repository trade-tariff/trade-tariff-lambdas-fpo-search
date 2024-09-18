import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import DescriptionLower

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = DescriptionLower()


class TestDescriptionLower(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        (
            ("99999999", "SIZE S"),
            ("99999999", "size s"),
        )
    ]

    def test_filter(self):
        for example, expected in TestDescriptionLower.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
