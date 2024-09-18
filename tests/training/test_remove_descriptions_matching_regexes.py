import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import RemoveDescriptionsMatchingRegexes

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = RemoveDescriptionsMatchingRegexes.build()


class TestRemoveDescriptionsMatchingRegexes(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        # skips descriptions with only numbers
        (
            ("15043090", "5234"),
            (None, None),
        ),
        # skips descriptions with only numbers and dashes
        (
            ("15043090", "5234-"),
            (None, None),
        ),
        # skips descriptions with only periods and slashes
        (
            ("15043090", "./"),
            (None, None),
        ),
        # skips descriptions with hyphens in between
        (
            ("15043090", "5234-5234"),
            (None, None),
        ),
        # skips descriptions with only numbers and asterisks
        (
            ("15043090", "5234*"),
            (None, None),
        ),
        # skips descriptions with just decimal numbers
        (
            ("15043090", "5234.5234"),
            (None, None),
        ),
        # skips descriptions with one or more digits and one or more whitespace characters
        (
            ("15043090", "5234 5234"),
            (None, None),
        ),
        # skips descriptions with only numbers and commas
        (
            ("15043090", "5234,"),
            (None, None),
        ),
        # keeps descriptions that do not match any regex
        (
            ("15043090", "This is a description"),
            ("15043090", "This is a description"),
        ),
    ]

    def test_filter(self):
        for example, expected in TestRemoveDescriptionsMatchingRegexes.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
