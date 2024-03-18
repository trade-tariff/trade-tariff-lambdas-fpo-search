import unittest
import logging
from unittest import mock

from data_sources.search_references import SearchReferences

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Test_search_references(unittest.TestCase):
    data_source = SearchReferences(url="http://localhost/test_search_references.json")

    response = {
        "data": [
            {
                "attributes": {
                    "referenced_class": "Commodity",
                    "negated_title": "Fresh Fish",
                    "goods_nomenclature_item_id": "12345",
                }
            }
        ]
    }

    def mock_response(self):
        mock_resp = mock.Mock()
        mock_resp.raise_for_status.side_effect = None
        return mock_resp

    @mock.patch("requests.get", side_effect=mock_response)
    def test_get_commodity_code(self, _mock_get):
        self.data_source._get = lambda: self.response
        self.assertEqual(self.data_source.get_commodity_code("fresh fish"), "12345")

    @mock.patch("requests.get", side_effect=mock_response)
    def test_includes_description_when_exists_is_true(self, _mock_get):
        self.data_source._get = lambda: self.response
        self.assertTrue(self.data_source.includes_description("FRESH FISH"))

    @mock.patch("requests.get", side_effect=mock_response)
    def test_includes_description_when_non_existant_is_false(self, _mock_get):
        self.data_source._get = lambda: self.response
        self.assertFalse(self.data_source.includes_description("FOO BAR"))


if __name__ == "__main__":
    unittest.main()
