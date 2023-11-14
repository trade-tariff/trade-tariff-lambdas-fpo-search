from os import PathLike
from typing import Union
from data_sources.data_source import DataSource
from lib.read_description_csv import read_description_csv


class TradeTariffDataSource(DataSource):
    def __init__(self, filename: Union[str, PathLike]) -> None:
        super().__init__()
        self._filename = filename

    def get_codes(self, digits: int) -> dict[str, list[str]]:
        code_data = read_description_csv(self._filename)
        mapped_codes = {}
        # build map of lines by code first, in case some are out of order
        for line in code_data:
            mapped_codes[line[9]] = line

        commodities_descriptions = {}

        # go back through the codes, building a text string from all the ancestors for each
        processed_codes = 0
        ignored_codes = 0
        for line in code_data:
            # only index the "end line" entries, for now? // TODO <- revisit this decision
            if line[6] != "1":
                ignored_codes += 1
                continue

            processed_codes += 1

            code = line[9]
            ancestor_codes = line[10].split(",")
            description = line[7]
            for ancestor_code in ancestor_codes:
                ancestor = mapped_codes[ancestor_code]
                description = ancestor[7] + " " + description

            key = code[:digits]
            if key in commodities_descriptions:
                commodities_descriptions[key].add(description)
            else:
                commodities_descriptions[key] = {description}

        print(f"Processed {processed_codes} codes, ignored {ignored_codes} codes")

        return commodities_descriptions

    def get_description(self) -> str:
        return f"Trade Tariff Descriptions from {str(self._filename)}"
