import logging
import unittest

from training.synonym.synonym_expander import SynonymExpander

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Test_synonym_expander(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terms_to_tokens = {
            "testing empty tokens": set(),
            "testingemptytokens": set(),
            "abridgement": {
                "abridgement",
                "abridgment",
            },
            "condensation": {"abridgment", "capsule", "condensation", "abridgement"},
            "abyssinian": {"abyssinian", "cat"},
            "yellow lemon": {"fruit"},
            "lemon": {"citrus fruit"},
            "red kite": {"alarming", "bird of prey"},
            "test": {"spaces"},
        }

    def test_expand_single_word_match(self):
        query = "abyssinian"
        expander = SynonymExpander(self.terms_to_tokens)
        expected = "abyssinian cat"
        self.assertEqual(expander.expand(query), expected)

    def test_expand_multi_word_match(self):
        query = "red kite"
        expander = SynonymExpander(self.terms_to_tokens)
        expected = "alarming bird of prey"
        self.assertEqual(expander.expand(query), expected)

    def test_expand_no_match(self):
        query = "pineapple"
        expander = SynonymExpander(self.terms_to_tokens, query)
        expected = "pineapple"
        self.assertEqual(expander.expand(query), expected)

    def test_expand_multi_token_match(self):
        query = "something about an abridgement"
        expander = SynonymExpander(self.terms_to_tokens, query)
        expected = "something about an abridgement abridgment"

        self.assertEqual(expander.expand(query), expected)

    def test_expand_phrase_and_word_collision(self):
        query = "yellow lemon is delicious"
        expander = SynonymExpander(self.terms_to_tokens, query)
        expected = "fruit is delicious citrus fruit"

        self.assertEqual(expander.expand(query), expected)

    def test_expand_empty_query(self):
        query = ""
        expander = SynonymExpander(self.terms_to_tokens, query)
        expected = ""

        self.assertEqual(expander.expand(query), expected)

    def test_empty_tokens_phrase(self):
        query = "testing empty tokens"
        expander = SynonymExpander(self.terms_to_tokens, query)
        expected = "testing empty tokens"

        self.assertEqual(expander.expand(query), expected)

    def test_empty_tokens_word(self):
        query = "testingemptytokens"
        expander = SynonymExpander(self.terms_to_tokens, query)
        expected = "testingemptytokens"

        self.assertEqual(expander.expand(query), expected)

    def test_expand_with(self):
        with SynonymExpander(self.terms_to_tokens, "red kite") as expanded_query:
            expected = "alarming bird of prey"

            self.assertEqual(expanded_query, expected)


if __name__ == "__main__":
    unittest.main()
