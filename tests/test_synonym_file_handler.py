import unittest
from training.synonym.synonym_file_handler import SynonymFileHandler


class TestSynonymFileHandler(unittest.TestCase):
    def test_load(self):
        handler = SynonymFileHandler(filename="tests/fixtures/test_synonyms.txt")
        expected_terms_to_tokens = {
            "abridgement": {
                "cat",
                "condensation",
                "abridgement",
                "abyssinian",
                "abridgment",
                "capsule",
            },
            "abridgment": {"abridgement", "condensation", "abridgment", "capsule"},
            "capsule": {"abridgement", "condensation", "abridgment", "capsule"},
            "condensation": {"abridgement", "condensation", "abridgment", "capsule"},
            "abyssinian": {"cat", "abyssinian"},
            "red lemon": {"bird"},
            "lemon": {"lime", "lemon", "citrus"},
            "lime": {"lime", "lemon", "citrus"},
            "citrus": {"lime", "lemon", "citrus"},
            "iron ore": {"iron ore", "hematite"},
            "hematite": {"iron ore", "hematite"},
            "fresh fish": {"fresh fish", "raw"},
            "raw": {"fresh fish", "raw"},
            "ocean crustations": {"ocean crustations", "shelled"},
            "shelled": {"ocean crustations", "shelled"},
            "long description": {"long description", "lengthy"},
            "lengthy": {"long description", "lengthy"},
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
                    "condensation",
                    "cat",
                    "abridgement",
                    "abyssinian",
                    "abridgment",
                },
                "abridgment": {"capsule", "abridgement", "abridgment", "condensation"},
                "capsule": {"capsule", "abridgement", "abridgment", "condensation"},
                "condensation": {
                    "capsule",
                    "abridgement",
                    "abridgment",
                    "condensation",
                },
                "abyssinian": {"cat", "abyssinian"},
                "red lemon": {"bird"},
                "lemon": {"lemon", "citrus", "lime"},
                "lime": {"lemon", "citrus", "lime"},
                "citrus": {"lemon", "citrus", "lime"},
                "iron ore": {"hematite", "iron ore"},
                "hematite": {"hematite", "iron ore"},
                "fresh fish": {"raw", "fresh fish"},
                "raw": {"raw", "fresh fish"},
                "ocean crustations": {"ocean crustations", "shelled"},
                "shelled": {"ocean crustations", "shelled"},
                "long description": {"long description", "lengthy"},
                "lengthy": {"long description", "lengthy"},
            }
            self.assertEqual(terms_to_tokens, expected_terms_to_tokens)


if __name__ == "__main__":
    unittest.main()
