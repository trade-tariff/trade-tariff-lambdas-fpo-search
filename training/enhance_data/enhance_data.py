from training.synonym.synonym_file_handler import SynonymFileHandler
from training.synonym.synonym_expander import SynonymExpander


class EnhanceData:
    def __init__(self, filename=None):
        self.filename = filename
        self.synonym_expander = self._synonym_expander()

    def add_synonyms(self, data):
        for code, descriptions in data.items():
            for index, description in enumerate(descriptions):
                descriptions[index] = self.synonym_expander.expand(description)

        return data

    def _synonym_expander(self):
        synonym_handler = SynonymFileHandler(filename=self.filename)
        synonym_handler.load()

        return SynonymExpander(synonym_handler.terms_to_tokens)
