import requests


class SearchReferences:
    SEARCH_REFS_API_URL = (
        "https://staging.trade-tariff.service.gov.uk/api/v2/search_references"
    )

    def __init__(self, url=SEARCH_REFS_API_URL):
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

    def _get(self):
        response = requests.get(self.url)

        return response.json()
