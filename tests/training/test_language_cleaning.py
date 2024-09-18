import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import LanguageCleaning
from train_args import TrainScriptArgsParser

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

args = TrainScriptArgsParser()
args.load_config_file()

language_skips_file = args.pwd() / args.partial_non_english_terms()
language_keeps_file = args.pwd() / args.partial_english_terms()
language_keeps_exact_file = args.pwd() / args.exact_english_terms()

with open(language_skips_file, "r") as f:
    language_skips = f.read().splitlines()

with open(language_keeps_file, "r") as f:
    language_keeps = f.read().splitlines()

with open(language_keeps_exact_file, "r") as f:
    language_keeps_exact = f.read().splitlines()

filter = LanguageCleaning(
    detected_languages=args.detected_languages(),
    preferred_languages=args.preferred_languages(),
    partial_skips=language_skips,
    partial_keeps=language_keeps,
    exact_keeps=language_keeps_exact,
)


class TestLanguageCleaning(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        ("faux fur", "faux fur"),
        (
            "deutsch influenza",
            "deutsch influenza",
        ),
        (
            "something english",
            "something english",
        ),
        (
            "something kinder",
            None,
        ),  # Non-English partial terms in descriptions are skipped
        ("espagna", None),  # detect Spanish descriptions and skip these
        ("deutsch", None),  # detect German descriptions and skip these
        ("francais", None),  # detect French descriptions and skip these
    ]

    def test_filter(self):
        for example, expected in TestLanguageCleaning.EXAMPLES:
            _, actual, _meta = filter.filter("subheading", example) or (None, None, {})
            self.assertEqual(actual, expected)
