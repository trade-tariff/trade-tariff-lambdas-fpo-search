import re
from typing import List
from data_sources.data_source import DataSource


class StaticDataSource(DataSource):
    def __init__(
        self,
        data: list[tuple[str, str]],
        authoritative: bool = False,
        creates_codes: bool = False,
        multiplier: int = 1,
    ):
        super().__init__(
            description="Static Data Source",
            authoritative=authoritative,
            creates_codes=creates_codes,
            multiplier=multiplier,
        )
        self._data = data

    def get_codes(self, digits: int) -> dict[str, List[str]]:
        codes = {}

        for description, subheading in self._data:
            subheading = subheading.replace(" ", "")[:digits]
            description = description.strip().lower()

            # Throw out any bad codes
            if not re.search("^\\d{" + str(digits) + "}$", subheading):
                continue

            if not description.strip():
                continue

            if subheading in codes:
                codes[subheading].add(description)
            else:
                codes[subheading] = {description}

        return codes
