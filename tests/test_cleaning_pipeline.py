import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import NegationCleaning

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = NegationCleaning.build()


class Test_NegationCleaning(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        ("some text, not other text", "some text"),
        ("some text, neither other text", "some text"),
        ("some text, other than other text", "some text"),
        ("some text, excluding other text", "some text"),
        ("some text, except other text", "some text"),
        (
            "shorts (other than swimwear) - women's or girl's knitted or crocheted",
            "shorts  - women's or girl's knitted or crocheted",
        ),
        (
            "fabric (textile) woven - vegetable textile fibres - (other than cotton and flax)",
            "fabric (textile) woven - vegetable textile fibres -",
        ),
        (
            "fabrics (textile) other than knitted, crocheted or woven - felt",
            "fabrics (textile)- felt",
        ),
        ("I have a\u00A0non-breaking space", "i have a non-breaking space"),
        ("some text", "some text"),
        (None, ""),
        ("", ""),
        (
            "some text, not other text.\nsome text, other than other text.",
            "some text\nsome text",
        ),
        (
            "Live cattle of a weight <= 80 kg (excl. pure-bred for breeding)",
            "live cattle of a weight <= 80 kg",
        ),
        ("Cereal grains, not otherwise worked than kibbled", "cereal grains"),
        ("Livers, other than fatty livers", "livers"),
        (
            "Fish, fresh or chilled, excluding fish fillets and other fish meat of heading 0304",
            "fish, fresh or chilled",
        ),
    ]

    def test_negation_cleaning(self):
        for example, expected in Test_NegationCleaning.EXAMPLES:
            _, actual, meta = filter.filter("subheading", example) or (None, None, {})
            self.assertEqual(actual, expected)
