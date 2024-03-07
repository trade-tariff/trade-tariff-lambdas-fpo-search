import unittest
from training.enhance_data.enhance_data import EnhanceData


class TestEnhanceData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {
            "12345678": {"fresh fish", "ocean crustations"},
            "55555555": {"iron ore"},
            "12345374": {"lemon"},
            "12237940": {
                "This is a long description that wont have the word added because its over one hundred characters length"
            },
            "12237890": {
                "This is a long description that will have a word added because it is shorter"
            },
        }

    def test_add_synonyms(self):
        enhance_data = EnhanceData(filename="tests/fixtures/test_synonyms.txt")

        expected_data = {
            "12345678": {"fresh fish raw", "ocean crustations shelled"},
            "55555555": {"iron ore hematite"},
            "12345374": {"lemon citrus lime"},
            "12237940": {
                "This is a long description that wont have the word added because its over one hundred characters length"
            },
            "12237890": {
                "This is a lengthy long description that will have word added because it shorter"
            },
        }
        self.assertEqual(enhance_data.add_synonyms(self.__class__.data), expected_data)


if __name__ == "__main__":
    unittest.main()
