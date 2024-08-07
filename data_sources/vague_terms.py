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
        self._codes: dict[str, set[str]] | None = None

    def get_codes(self, digits: int) -> dict[str, set[str]]:
        with open(self._filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            code_data = list(csv_reader)

        codes = {}

        for line in code_data:
            description = line[0].strip().lower()

            if vague_term_code in codes:
                codes[vague_term_code].add(description)
            else:
                codes[vague_term_code] = {description}

        self._codes = codes

        return codes

    def includes_description(self, description) -> bool:
        if self._codes is None:
            self.get_codes(0)

        if self._codes:
            return description.strip().lower() in self._codes[vague_term_code]
        else:
            return False
