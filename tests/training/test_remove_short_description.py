import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import RemoveShortDescription

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = RemoveShortDescription(min_length=4)


class TestRemoveShortDescription(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        (
            ("15043090", "HAIR"),
            (None, None),
        ),
        (
            ("15043090", "HAIRY"),
            ("15043090", "HAIRY"),
        ),
    ]

    def test_filter(self):
        for example, expected in TestRemoveShortDescription.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
