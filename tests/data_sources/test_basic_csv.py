import logging
import unittest

from data_sources.basic_csv import BasicCSVDataSource

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Test_basic_csv_data_source(unittest.TestCase):
    sample_file_path = "tests/data_sources/sample_data.csv"
    data_source = BasicCSVDataSource(filename=sample_file_path)

    def test_initialization(self):
        self.assertEqual(self.data_source._filename, self.sample_file_path)
        self.assertEqual(self.data_source._code_col, 0)
        self.assertEqual(self.data_source._description_col, 1)
        self.assertEqual(self.data_source._encoding, "utf-8")

    def test_get_codes(self):
        result = self.data_source.get_codes(digits=5)
        expected = {
            "12345": {"fresh fish", "raw ocean fish"},
            "23456": {"dried fruit"},
            "34567": {"wooden frames"},
            "88888": {"plastic toys"},
        }
        self.assertEqual(result, expected)

    def test_get_description(self):
        expected_description = f"CSV data source from {self.sample_file_path}"
        self.assertEqual(self.data_source.description, expected_description)


if __name__ == "__main__":
    unittest.main()
