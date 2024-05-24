import csv
from os import PathLike
import re
from typing import Union
from data_sources.data_source import DataSource


class BasicCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
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

            # Throw out any bad codes
            if not re.search("^\\d{" + str(digits) + "}$", subheading):
                continue

            if not description.strip():
                continue #Throw out any blank descriptions

            if len(description) <= 4:  # Skip descriptions with 4 or less characters
                continue
            
            if re.search(r"^\\d+$", description): ## Skip if the description consists entirely of digits only
                continue 
            if re.search(r'^[0-9-]+$', description): # Skip rows where description contains only numbers and dashes
                continue
            if re.search(r'^[./]+$', description): # Skip rows where description consists only of a '.' or a '/'
                continue
            if re.search(r"^\d+-\d+$", description): #skip numbers with hyphens in between
                continue
            if re.search(r'^[0-9*]+$', description): # Skip rows where description contains only numbers and asterisks
                continue
            if re.search(r"^[-+]?\d+(\.\d+)?$", description): ##skip if just decimal numbers
                continue
            if re.search(r'^\d+\s+\d+$', description): # Skip rows where description contains one or more digits and one or more whitespace characters (including spaces, tabs, and other Unicode spaces)
                continue
            if re.search(r'^[0-9,]+$', description): # Skip rows where description contains only numbers and commas
                continue

            if subheading in codes:
                codes[subheading].add(description)
            else:
                codes[subheading] = {description}

        return codes
