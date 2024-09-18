import logging
from typing import List, Tuple
import unittest

from training.cleaning_pipeline import PhraseRemover
from train_args import TrainScriptArgsParser

args = TrainScriptArgsParser()
args.load_config_file()
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

filter = PhraseRemover.build(args.phrases_to_remove_file())


class TestPhraseRemover(unittest.TestCase):
    EXAMPLES: List[Tuple] = [
        (
            ("99999999", "gold no commercial value"),
            ("99999999", "gold")
        ),
        (
            ("99999999", "no commercial value"),
            (None, None)
        )
    ]

    def test_filter(self):
        for example, expected in TestPhraseRemover.EXAMPLES:
            example_subheading, example_description = example
            expected_subheading, expected_description = expected

            actual_subheading, actual_description, _meta = filter.filter(
                example_subheading, example_description
            )

            self.assertEqual(actual_subheading, expected_subheading)
            self.assertEqual(actual_description, expected_description)
