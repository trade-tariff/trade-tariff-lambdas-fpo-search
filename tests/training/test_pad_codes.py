import logging
import unittest
from typing import List, Tuple

from training.cleaning_pipeline import PadCodes

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = PadCodes.build()


class TestPadCodes(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        ("01010100", "01010100"),
        ("010101", "01010100"),
        ("0101", "0101"),
        ("", ""),
    ]

    def test_filter(self):
        for example, expected in TestPadCodes.EXAMPLES:
            actual, _, _ = filter.filter(example, "irrelevant") or (None, None, {})
            self.assertEqual(actual, expected)
