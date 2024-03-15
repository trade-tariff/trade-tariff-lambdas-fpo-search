import logging
import unittest
from unittest import mock

from data_sources.basic_csv import BasicCSVDataSource
from data_sources.search_references import SearchReferences

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Test_basic_csv_data_source(unittest.TestCase):
    sample_file_path = "tests/data_sources/sample_data.csv"
    search_references = SearchReferences(
        url="http://localhost/test_search_references.json"
    )
    data_source = BasicCSVDataSource(
        filename=sample_file_path,
        search_references=search_references,
    )

    response = {
        "data": [
            {
                "attributes": {
                    "referenced_class": "Commodity",
                    "negated_title": "Fresh Fish",
                    "goods_nomenclature_item_id": "12345",
                }
            },
            {
                "attributes": {
                    "referenced_class": "Commodity",
                    "negated_title": "Plastic Toys",
                    "goods_nomenclature_item_id": "39269",
                }
            },
        ]
    }

    def mock_response(self):
        mock_resp = mock.Mock()
        mock_resp.raise_for_status.side_effect = None
        return mock_resp

    def test_initialization(self):
        self.assertEqual(self.data_source._filename, self.sample_file_path)
        self.assertEqual(self.data_source._code_col, 0)
        self.assertEqual(self.data_source._description_col, 1)
        self.assertEqual(self.data_source._encoding, "utf-8")

    @mock.patch("requests.get", side_effect=mock_response)
    def test_get_codes(self, _mock_get):
        self.search_references._get = lambda: self.response
        result = self.data_source.get_codes(digits=5)
        expected = {
            "12345": {"fresh fish", "raw ocean fish"},
            "23456": {"dried fruit"},
            "34567": {"wooden frames"},
            "39269": {"plastic toys"},
        }
        self.assertEqual(result, expected)

    def test_get_description(self):
        expected_description = f"CSV data source from {self.sample_file_path}"
        self.assertEqual(self.data_source.get_description(), expected_description)


if __name__ == "__main__":
    unittest.main()
