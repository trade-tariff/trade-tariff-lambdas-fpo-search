import csv
import os
from os import PathLike
import re
from typing import Union
from data_sources.data_source import DataSource
import pandas as pd

DEFAULT_SEARCH_REFERENCES_FILE = (
    "raw_source_data/tradesets_descriptions/search_references_final8digit_16thFeb.csv"
)


class BasicCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
        code_col: int = 0,
        description_col: int = 1,
        search_references_file: Union[str, PathLike] = DEFAULT_SEARCH_REFERENCES_FILE,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self._filename = filename
        self._code_col = code_col
        self._description_col = description_col
        self._encoding = encoding

        if os.path.isfile(search_references_file):
            self._search_refs = pd.read_csv(search_references_file, dtype=str)
        else:
            self._search_refs = None

    def get_codes(self, digits: int) -> dict[str, list[str]]:
        with open(self._filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # skip the first line (header)
            code_data = list(csv_reader)

        codes = {}

        count = 0
        for row in code_data:
            subheading = row[self._code_col].strip()[:digits]
            description = row[self._description_col].strip().lower()

            # Throw out any bad codes
            if not re.search("^\\d{" + str(digits) + "}$", subheading):
                continue

            if self._search_refs is not None:
                # Check if the description exists in search references
                if description in self._search_refs["GDSDESC"].values:
                    count += 1

                    # Find the corresponding CMDTYCODE for the description
                    cmdtycode = (
                        self._search_refs.loc[
                            self._search_refs["GDSDESC"] == description, "CMDTYCODE"
                        ]
                        .iloc[0]
                        .strip()[:digits]
                    )
                    row[
                        self._code_col
                    ] = cmdtycode  # replace the values in self._code_col with the corresponding CMDTYCODE from the mapping_dict if the description matches
                    subheading = cmdtycode

            if subheading in codes:
                codes[subheading].add(description)
            else:
                codes[subheading] = {description}

        print(f"Count of matches: {count}")

        return codes

    def get_description(self) -> str:
        return f"CSV data source from {str(self._filename)}"
