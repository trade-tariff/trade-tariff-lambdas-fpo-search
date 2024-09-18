import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import IncorrectPairsRemover
from train_args import TrainScriptArgsParser

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

args = TrainScriptArgsParser()
args.load_config_file()
filter = IncorrectPairsRemover.build(args.incorrect_description_pairs_file())


class TestIncorrectPairsRemover(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        # skip known incorrect code pairs
        (
            ("15043090", "HAIR STRAIGHTENER"),
            (None, None),
        ),
        # skip known incorrect chapter pairs
        (
            ("11111111", "MACBOOK"),
            (None, None),
        ),
        # preserve known correct chapter
        (
            ("84111111", "MACBOOK"),
            ("84111111", "MACBOOK"),
        ),
    ]

    def test_filter(self):
        for example, expected in TestIncorrectPairsRemover.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
