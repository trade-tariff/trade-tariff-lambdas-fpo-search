import spacy
from spellchecker import SpellChecker
import csv

"""
This script is a spike to test the spellchecker and the spacy library.
The idea is to use spacy library to identify the NOUNS in the text and then use the spellchecker to correct them.
The results are still pretty poor,
unfortunatelly the missing context is a big problem and the spellchecker is not able to correct the words properly.
A different approach would be using a LLM model to correct them.

To run this script:
python aws_lambda/contextual_corrector.py
"""


class NounsSpellingCorrector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.spell_checker = SpellChecker(distance=1)

    def correct(self, text):
        doc = self.nlp(text)
        return doc._.outcome_spellCheck

    def suggestions(self, text):
        doc = self.nlp(text)
        return doc._.suggestions_spellCheck


if __name__ == "__main__":
    cc = NounsSpellingCorrector()

    # Load data from CSV file:
    csv_file_path = (
        "./raw_source_data/tradesets_descriptions/DEC22COMCODEDESCRIPTION.csv"
    )
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)

        for line in csv_reader:
            text = line[1]

            doc = cc.nlp(text)

            corrected = []

            for token in doc:
                if token.pos_ == "NOUN":
                    tocken_corrected = cc.spell_checker.correction(token.text.strip())

                    if tocken_corrected is not None:
                        if tocken_corrected != token.text:
                            print(text)
                            print(f"{token.text}->{tocken_corrected}")

                        corrected.append(tocken_corrected)
                    else:
                        corrected.append(token.text)
                else:
                    corrected.append(token.text)

            corrected = [word for word in corrected if word is not None]

            print(" ".join(corrected))

            if corrected != token.text:
                print(" --- ")
