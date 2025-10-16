import csv
import os
import re
from os import PathLike
from typing import Union

from data_sources.data_source import DataSource
from inference.infer import vague_term_code


class VagueTermsCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
        patterns_file: Union[str, PathLike],
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
        self.filename = filename
        self._patterns_file = patterns_file
        self._encoding = encoding
        self._codes: dict[str, set[str]] = {}
        self._patterns: list[re.Pattern] = []

    def get_codes(self, digits: int) -> dict[str, set[str]]:
        with open(self.filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            code_data = list(csv_reader)

        for line in code_data:
            description = line[0].strip().lower()

            if vague_term_code in self._codes:
                self._codes[vague_term_code].add(description)
            else:
                self._codes[vague_term_code] = {description}

        return self._codes

    def get_patterns(self) -> None:
        if os.path.exists(self._patterns_file):
            with open(self._patterns_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            self._patterns.append(re.compile(line, re.IGNORECASE))
                        except re.error:
                            print(
                                f"Warning: Invalid regex in {self._patterns_file}: {line}"
                            )

    def includes_description(self, description) -> bool:
        description = description.strip().lower()

        if not self._codes:
            self.get_codes(0)

        if not self._patterns:
            self.get_patterns()

        if description in self._codes[vague_term_code]:
            return True

        if any(pattern.match(description) for pattern in self._patterns):
            return True

        return False
