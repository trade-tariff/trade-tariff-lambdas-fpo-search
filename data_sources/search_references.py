import re
import requests

from data_sources.data_source import DataSource


class SearchReferencesDataSource(DataSource):
    SEARCH_REFS_API_URL = (
        "https://staging.trade-tariff.service.gov.uk/api/v2/search_references"
    )

    def __init__(
        self,
        url=SEARCH_REFS_API_URL,
        authoritative: bool = True,
        creates_codes: bool = False,
        multiplier: int = 1,
    ):
        super().__init__(
            description=f"Search references data source from {str(url)}",
            authoritative=authoritative,
            creates_codes=creates_codes,
            multiplier=multiplier,
        )

        self.url = url
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
                commodities[
                    entry["attributes"]["negated_title"].strip().lower()
                ] = entry["attributes"]["goods_nomenclature_item_id"]

        self._commodities = commodities
        return commodities

    def get_codes(self, digits: int) -> dict[str, list[str]]:
        commodities = self.commodities()

        documents = {}

        for description, code in commodities.items():
            subheading = code.strip()[:digits]

            # Throw out any bad codes
            if not re.search("^\\d{" + str(digits) + "}$", subheading):
                continue

            if subheading in documents:
                documents[subheading].add(description)
            else:
                documents[subheading] = {description}

        return documents

    def _get(self):
        response = requests.get(self.url)

        return response.json()
