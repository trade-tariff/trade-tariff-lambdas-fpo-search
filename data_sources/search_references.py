import requests
import logging
import json

from data_sources.data_source import DataSource

logger = logging.getLogger(__name__)


class SearchReferencesDataSource(DataSource):
    SEARCH_REFS_API_URL = (
        "https://staging.trade-tariff.service.gov.uk/api/v2/search_references"
    )
    DEFAULT_PATH = "reference_data/search_references.json"

    def __init__(
        self,
        url=SEARCH_REFS_API_URL,
        authoritative: bool = True,
        creates_codes: bool = False,
        multiplier: int = 1,
        json_codes: dict[str, str] | None = None,
    ):
        super().__init__(
            description=f"Search references data source from {str(url)}",
            authoritative=authoritative,
            creates_codes=creates_codes,
            multiplier=multiplier,
        )

        self.url = url
        if json_codes is not None:
            self._commodities = json_codes
        else:
            self._commodities = None

    def get_commodity_code(self, description):
        commodity_code = None

        if self.includes_description(description):
            commodity_code = self.commodities()[description.strip().lower()]

        return commodity_code

    def includes_description(self, description):
        return description.strip().lower() in self.commodities()

    def commodities(self):
        if self._commodities is not None:
            return self._commodities

        response = self._get()

        json_entries = response["data"]

        commodities = {}
        for entry in json_entries:
            if entry["attributes"]["referenced_class"] in ["Commodity", "Subheading"]:
                commodities[entry["attributes"]["negated_title"].strip().lower()] = (
                    entry["attributes"]["goods_nomenclature_item_id"]
                )

        self._commodities = commodities
        return commodities

    def get_codes(self, digits: int) -> dict[str, set[str]]:
        commodities = self.commodities()

        documents = {}

        for description, code in commodities.items():
            subheading = code.strip()[:digits]

            if subheading in documents:
                documents[subheading].add(description)
            else:
                documents[subheading] = {description}

        total_descriptions = sum(
            len(descriptions) for descriptions in documents.values()
        )
        unique_descriptions = len(
            {
                description
                for descriptions in documents.values()
                for description in descriptions
            }
        )
        logger.info(
            f"Loaded {len(documents)} subheadings with {unique_descriptions} unique descs and {total_descriptions} total descs"
        )
        return documents

    def write_as_json(self, path: str | None = None):
        path = path or self.DEFAULT_PATH

        with open(path, "w") as f:
            f.write(
                json.dumps(
                    self.commodities(),
                    indent=4,
                )
            )

    @classmethod
    def build_from_json(cls, path: str | None = None) -> "SearchReferencesDataSource":
        path = path or cls.DEFAULT_PATH

        with open(path) as f:
            json_content = json.load(f)

            return cls(json_codes=json_content)

    def _get(self):
        response = requests.get(self.url)

        return response.json()
