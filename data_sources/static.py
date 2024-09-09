import re
from typing import List
from data_sources.data_source import DataSource
import pandas as pd

# Phrases to remove
phrases_to_remove = ["value for customs purpose only", "for customs purposes only", "no commercial value declared for customs purposes only", "no commercial value"
                     ,"rtn to shipper never enter us commerce", "for customs purpose only", "(value for customs purpose only)"]

pattern = '|'.join(map(re.escape, phrases_to_remove))

incorrect_code_desc_pairs=pd.read_csv('Definitely Incorrect Code Desc Pairs_Final.csv', dtype=str)

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
            description = (re.sub(pattern, '', description)) ##remove certain phrases, if they match the above phrases_to_remove list
            description = description.rstrip(',').rstrip('.') #remove trailing commas #remove trailing full stops (add to line above for speed)

            # Throw out any bad codes
            if not re.search("^\\d{" + str(digits) + "}$", subheading):
                continue

            if not description.strip():
                continue

            #Throw out any descriptions with excessively long words (unless from chemicals section??)
            #words = description.split()
            #if any(len(word) > 45 for word in words):
            #    continue

            #incorrect pairs:
            if description in incorrect_code_desc_pairs['GDSDESC'].values:
                # Find the corresponding CMDTYCODE for the description
                cmdtycode_seen = incorrect_code_desc_pairs.loc[incorrect_code_desc_pairs['GDSDESC'].str.lower() == description, 'CMDTYCODE_Seen'].values[0]
                chapter_shouldbe = incorrect_code_desc_pairs.loc[incorrect_code_desc_pairs['GDSDESC'].str.lower() == description, 'Chapter_ShouldBe'].values[0]
                #line[self._code_col] = cmdtycode[:digits] #replace the values in self._code_col with the corresponding CMDTYCODE from the mapping_dict if the description matches
                if subheading[:2] == cmdtycode_seen[:2]:
                    #print(f"skipping seen: {description, subheading}")
                    continue
                if not pd.isna(chapter_shouldbe) and chapter_shouldbe !='': 
                    if subheading[:2] != chapter_shouldbe:
                        #print(f"skipping: {description, subheading, chapter_shouldbe}")
                        continue

            #Join trailing s unless preceded by word 'size'
            description = re.sub(r'\bsize\s+s\b', 'size_placeholder', description)
            pattern2 = r'\b(\w+)\s+s\b'
            description = re.sub(pattern2, r'\1s', description)
            description = re.sub(r'size_placeholder', 'size s', description)

            if subheading in codes:
                codes[subheading].add(description)
            else:
                codes[subheading] = {description}

        return codes
