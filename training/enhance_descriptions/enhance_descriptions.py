from training.synonym.synonym_file_handler import SynonymFileHandler
from training.synonym.synonym_expander import SynonymExpander
from training.helpers import Helpers


class EnhanceDescriptions:
    def __init__(self, filename=None):
        self.filename = filename
        self._synonym_expander = None

    @property
    def synonym_expander(self):
        if self._synonym_expander is None:
            self._initialize_synonym_expander()
        return self._synonym_expander

    def _initialize_synonym_expander(self):
        """Lazily initializes the SynonymExpander for heavy processing or dependencies."""
        try:
            synonym_handler = SynonymFileHandler(filename=self.filename)
            synonym_handler.load()
            self._synonym_expander = SynonymExpander(synonym_handler.terms_to_tokens)
        except Exception as e:
            print(f"Failed to initialize synonym expander: {e}")
            raise

    def add_synonyms(self, data):
        """Expands synonyms in the given data, returning a new data structure to avoid mutation but in type."""
        new_data = {}

        for code, descriptions in data.items():
            expanded_descriptions = set()
            for description in descriptions:
                if len(description) < 100:
                    expanded_description = self.synonym_expander.expand(description)
                    expanded_descriptions.add(description)
                else:
                    expanded_description = description

                expanded_descriptions.add(Helpers.unique_words(expanded_description))

            new_data[code] = expanded_descriptions
        return new_data
