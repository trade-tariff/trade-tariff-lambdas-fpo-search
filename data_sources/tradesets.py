from os import PathLike
import re
from typing import Union
from data_sources.data_source import DataSource
from lib.read_description_csv import read_description_csv


class BasicCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
        code_col: int = 0,
        description_col: int = 1,
    ) -> None:
        super().__init__()
        self._filename = filename
        self._code_col = code_col
        self._description_col = description_col

    def get_codes(self, digits: int) -> dict[str, list[str]]:
        csv = read_description_csv(self._filename)

        documents = {}

        for line in csv:
            subheading = line[self._code_col].strip()[:digits]
            description = line[self._description_col].strip()

            # Throw out any bad codes
            if not re.search("^\\d{" + str(digits) + "}$", subheading):
                continue

            if subheading in documents:
                documents[subheading].add(description)
            else:
                documents[subheading] = {description}

        return documents

    def get_description(self) -> str:
        return f"CSV data source from {str(self._filename)}"
