import unittest
from training.synonym.synonym_file_handler import SynonymFileHandler


class TestSynonymFileHandler(unittest.TestCase):
    def test_load(self):
        handler = SynonymFileHandler(filename="tests/fixtures/test_synonyms.txt")
        expected_terms_to_tokens = {
            "abridgement": {
                "abridgment",
                "condensation",
                "abridgement",
                "capsule",
                "abyssinian",
                "cat",
            },
            "abridgment": {"abridgment", "condensation", "abridgement", "capsule"},
            "capsule": {"abridgment", "condensation", "abridgement", "capsule"},
            "condensation": {"abridgment", "condensation", "abridgement", "capsule"},
            "abyssinian": {"abyssinian", "cat"},
            "red lemon": {"bird"},
            "lemon": {"lime", "citrus", "lemon"},
            "lime": {"lime", "citrus", "lemon"},
            "citrus": {"lime", "citrus", "lemon"},
            "iron ore": {"raw", "iron ore"},
            "raw": {"raw", "fresh fish", "iron ore"},
            "fresh fish": {"raw", "fresh fish"},
            "ocean crustations": {"shelled", "ocean crustations"},
            "shelled": {"shelled", "ocean crustations"},
        }

        self.assertEqual(handler.load(), expected_terms_to_tokens)

    def test_load_empty_file(self):
        handler = SynonymFileHandler(filename="tests/fixtures/test_empty_synonyms.txt")
        expected_terms_to_tokens = {}
        self.assertEqual(handler.load(), expected_terms_to_tokens)

    def test_load_with(self):
        with SynonymFileHandler(
            filename="tests/fixtures/test_synonyms.txt"
        ) as terms_to_tokens:
            expected_terms_to_tokens = {
                "abridgement": {
                    "capsule",
                    "abridgement",
                    "abridgment",
                    "condensation",
                    "abyssinian",
                    "cat",
                },
                "abridgment": {"capsule", "abridgment", "condensation", "abridgement"},
                "capsule": {"capsule", "abridgment", "condensation", "abridgement"},
                "condensation": {
                    "capsule",
                    "abridgment",
                    "condensation",
                    "abridgement",
                },
                "abyssinian": {"abyssinian", "cat"},
                "red lemon": {"bird"},
                "lemon": {"citrus", "lemon", "lime"},
                "lime": {"citrus", "lemon", "lime"},
                "citrus": {"citrus", "lemon", "lime"},
                "iron ore": {"raw", "iron ore"},
                "raw": {"raw", "fresh fish", "iron ore"},
                "fresh fish": {"raw", "fresh fish"},
                "ocean crustations": {"ocean crustations", "shelled"},
                "shelled": {"ocean crustations", "shelled"},
            }
            print(terms_to_tokens)
            self.assertEqual(terms_to_tokens, expected_terms_to_tokens)


if __name__ == "__main__":
    unittest.main()
