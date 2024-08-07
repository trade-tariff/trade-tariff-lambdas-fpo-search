import unittest
import logging
import json
import os

from data_sources.search_references import SearchReferencesDataSource

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


class Test_search_references(unittest.TestCase):
    data_source = SearchReferencesDataSource(
        url="http://localhost/test_search_references.json"
    )

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

    def test_get_commodity_code(self):
        self.data_source._get = lambda: self.response
        self.assertEqual(self.data_source.get_commodity_code("fresh fish"), "12345")

    def test_includes_description_when_exists_is_true(self):
        self.data_source._get = lambda: self.response
        self.assertTrue(self.data_source.includes_description("FRESH FISH"))

    def test_includes_description_when_non_existant_is_false(self):
        self.data_source._get = lambda: self.response
        self.assertFalse(self.data_source.includes_description("FOO BAR"))

    def test_write_as_json(self):
        self.data_source._get = lambda: self.response
        self.data_source.write_as_json(path="test_search_references.json")

        try:
            with open("test_search_references.json") as f:
                content = f.read()
                json_content = json.loads(content)
                self.assertEqual(json_content, {"fresh fish": "12345"})
        finally:
            os.remove("test_search_references.json")

    def test_build_from_json(self):
        data_source = SearchReferencesDataSource.build_from_json()

        self.assertEqual(data_source.get_commodity_code("ricotta"), "0406105090")
