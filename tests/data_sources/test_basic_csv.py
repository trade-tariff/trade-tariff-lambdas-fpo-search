import logging
import unittest

from data_sources.basic_csv import BasicCSVDataSource

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Test_basic_csv_data_source(unittest.TestCase):
    def test_initialization(self):
        data_source = BasicCSVDataSource("test.csv")
        self.assertEqual(data_source._filename, "test.csv")
        self.assertEqual(data_source._code_col, 0)
        self.assertEqual(data_source._description_col, 1)
        self.assertEqual(data_source._encoding, "utf-8")

    def test_get_codes(self):
        sample_file_path = "tests/data_sources/sample_data.csv"
        data_source = BasicCSVDataSource(sample_file_path)

        result = data_source.get_codes(digits=5)
        expected = {
            "12345": {"Fresh fish", "Raw ocean fish"},
            "23456": {"Dried fruit"},
            "34567": {"Wooden frames"},
        }
        self.assertEqual(result, expected)

    def test_get_description(self):
        data_source = BasicCSVDataSource("test.csv")
        expected_description = "CSV data source from test.csv"
        self.assertEqual(data_source.get_description(), expected_description)


if __name__ == "__main__":
    unittest.main()
