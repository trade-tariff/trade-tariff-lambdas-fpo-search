import csv
from os import PathLike
from typing import Union, Optional
from data_sources.data_source import DataSource
from training.cleaning_pipeline import CleaningPipeline



class BasicCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
        cleaning_pipeline: Optional[CleaningPipeline] = None,
        code_col: int = 0,
        description_col: int = 1,
        encoding: str = "utf-8",
        authoritative: bool = False,
        creates_codes: bool = False,
        multiplier: int = 1,
    ) -> None:
        super().__init__(
            description=f"CSV data source from {str(filename)}",
            authoritative=authoritative,
            creates_codes=creates_codes,
            multiplier=multiplier,
            cleaning_pipeline=cleaning_pipeline,
        )
        self._filename = filename
        self._code_col = code_col
        self._description_col = description_col
        self._encoding = encoding

    def get_codes(self, digits: int) -> dict[str, list[str]]:
        with open(self._filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # skip the first line (header)
            code_data = list(csv_reader)

        codes = {}

        for row in code_data:
            subheading = row[self._code_col].replace(" ", "")[:digits]
            description = row[self._description_col].strip().lower()

            if self.cleaning_pipeline:
                result = self.cleaning_pipeline.filter(subheading, description)

                if result is None:
                    continue

                subheading, description = result

            if subheading in codes:
                codes[subheading].add(description)
            else:
                codes[subheading] = {description}

        return codes
