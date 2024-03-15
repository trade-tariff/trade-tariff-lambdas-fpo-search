import csv
from os import PathLike
import re
from typing import Union
from data_sources.data_source import DataSource
import pandas as pd


final_df=pd.read_csv('/home/ec2-user/SageMaker/trade-tariff-lambdas-fpo-search/raw_source_data/tradesets_descriptions/search_references_final8digit_16thFeb.csv', dtype=str)

class BasicCSVDataSource(DataSource):
    def __init__(
        self,
        filename: Union[str, PathLike],
        code_col: int = 0,
        description_col: int = 1,
        encoding: str = "Windows-1252", #utf-8
    ) -> None:
        super().__init__()
        self._filename = filename
        self._code_col = code_col
        self._description_col = description_col
        self._encoding = encoding

    def get_codes(self, digits: int) -> dict[str, list[str]]:
        with open(self._filename, mode="r", encoding=self._encoding) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # skip the first line (header)
            code_data = list(csv_reader)

        documents = {}

        for line in code_data:
            subheading = line[self._code_col].strip()[:digits]
            description = line[self._description_col].strip().lower()

            # Throw out any bad codes
            if not re.search("^\\d{" + str(digits) + "}$", subheading):
                continue

            #Throw out any blank descriptions if any remain
            if not description.strip():
                continue
                
            if description in final_df['GDSDESC'].values:
                # Find the corresponding CMDTYCODE for the description
                cmdtycode = final_df.loc[final_df['GDSDESC'] == description, 'CMDTYCODE'].values[0]
                line[self._code_col] = cmdtycode #replace the values in self._code_col with the corresponding CMDTYCODE from the mapping_dict if the description matches
                subheading=cmdtycode
            
            if subheading in documents:
                documents[subheading].add(description)
            else:
                documents[subheading] = {description}

        return documents

    def get_description(self) -> str:
        return f"CSV data source from {str(self._filename)}"
