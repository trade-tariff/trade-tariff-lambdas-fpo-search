import csv
from os import PathLike
import re
from typing import Union
from data_sources.data_source import DataSource
from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.7).build()

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

            specific_phrase=["pendant", "moccasins", "bandanas", "it equipment", "bacillus subtilis flagellin", "voltage stabilizer", "fabric"
                    , "primer", "lens"] #fabric vague?
            similar_words=["cannula", "cannula injection", "naproxen", "vial", "diffuser", "carburettor", "(iso)"," iso", "antigen", "dress", " ssd", "oil"
                ,"watch", "adapter", "hdmi", "cable", "kit", "diagnostic", "protein", "bacteria", "diagno", "rna", "allergen", "peptide"
                ,"test", "charger", "inverter", "laptop", "ipad", "tablet", "controller", "android", "samsung", "lenovo", "thinkpad"
                ,"primocin", "amino", "liquid", "notebook", "lingerie", "globulin", "influenza", "blood", "human", "animal", "nissan","phone"
                , "pure", "acid", "yl", "phos", "pyr", "thymidine", "nucleic", "oxide", "benz", "toshiba", "nokia", "chloro", "oxy", "hex"
                ,"kaftan", "chemise", "tracolimus", "bra", "footmuff", "magnifier", "photoframe", "kinder bueno", "kinder chocolate" 
                 , "kinderbox", "katjes kinder", "mens bag", "mens shirt", "mens vest", "mens mini bag", "mens polo", "mens cologne"
                 , "mens parka", "mens shirt", "mens linen", "mens clothing", "vest mens", "mens vest", "mens pant"
                 , "mens bomber", "mens tshirt", "mens trainer", "mens shoes", "mens belt", "mens cap", "mens hat"
                 , "mens denim", "mens hat", "mens tie", "mens anorak", "mens boot", "eau de parfum"]

            if authoritative=False:
                language = detector.detect_language_of(description)
                if description in specific_phrase:
                    # Check if any specified keywords are present in lowercase description
                    if any(keyword==description for keyword in specific_phrase):
                        description=description  # keep exact matches
                    elif any(keyword.lower() in description for keyword in similar_words):
                        description=description # keep if appears anywhere in string
                    ###Condition to remove 'kinder' but keep certain mentions of it (above)
                    elif 'kinder' in description:
                        continue ##remove mentions of kinder even if confidence is higher than 0.2 / assigned as eng or None (except specific word combos above)
                    elif language=='ENG' or language=='None':
                        description=description ##else only keep if assigned as english or None
                    else:
                        continue ##else don't keep
        

            
            if language == None:
            language_min_rel_distance_point7.append((text, 'None', label, file))
        else:
            language_min_rel_distance_point7.append((text,language.iso_code_639_3.name, label, file))

            

            if subheading in codes:
                codes[subheading].add(description)
            else:
                codes[subheading] = {description}

        return codes
