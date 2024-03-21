import csv
from os import PathLike
from typing import Union
from data_sources.data_source import DataSource
from inference.infer import vague_term_code


class VagueTermsCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
        encoding: str = "utf-8",
        authoritative: bool = True,
        creates_codes: bool = True,
        multiplier: int = 1,
    ) -> None:
        super().__init__(
            description=f"Vague terms data source from {str(filename)}",
            authoritative=authoritative,
            creates_codes=creates_codes,
            multiplier=multiplier,
        )
        self._filename = filename
        self._encoding = encoding

    def get_codes(self, digits: int) -> dict[str, list[str]]:
        with open(self._filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # skip the first line (header)
            code_data = list(csv_reader)

        documents = {}

        for line in code_data:
            subheading = vague_term_code
            description = line[0].strip().lower()

            if subheading in documents:
                documents[subheading].add(description)
            else:
                documents[subheading] = {description}

        return documents
