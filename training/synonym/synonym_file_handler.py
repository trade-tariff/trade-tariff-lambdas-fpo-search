"""
Converts synonym files into a dictionary of terms to tokens which can be used to expand queries
into a set of equivalent synonym tokens. Handles empty lines, whitespace, and duplicate tokens as well as equivalent
mappings (e.g. "abyssinian => cat, abyssinian") and explicit mappings (e.g. "abridgement, abridgment, capsule,
condensation").

Parameters:
    filename (str): The path to the synonym file.

Returns:
    dict: A dictionary of terms to tokens.
"""


class SynonymFileHandler:
    SYNONYM_FALLBACK_FILEPATH = "config/data/synonyms-all-fallback.txt"

    def __init__(self, filename=None):
        self.filename = filename
        self.terms_to_tokens = {}

    def load(self):
        self._parse_file()

        return self.terms_to_tokens

    def _parse_file(self):
        if self.filename:
            filename = self.filename
        else:
            filename = SynonymFileHandler.SYNONYM_FALLBACK_FILEPATH

        with open(filename, "r") as f:
            synonym_lines = f.read().splitlines()

            for line in synonym_lines:
                if not line:
                    continue
                else:
                    lhs, rhs = (
                        [r.strip() for r in line.split("=>")]
                        if "=>" in line
                        else (line, None)
                    )

                    # Explicit mappings
                    if rhs:
                        terms = [t.strip() for t in lhs.split(",")]
                        tokens = [r.strip() for r in rhs.split(",")]
                        for term in terms:
                            if term in self.terms_to_tokens:
                                self.terms_to_tokens[term] = self.terms_to_tokens[
                                    term
                                ].union(set(tokens))
                            else:
                                self.terms_to_tokens[term] = set(tokens)

                    # Equivalent mappings
                    elif lhs:
                        tokens = [t.strip() for t in lhs.split(",")]
                        for term in tokens:
                            if term in self.terms_to_tokens:
                                self.terms_to_tokens[term] = self.terms_to_tokens[
                                    term
                                ].union(set(tokens))
                            else:
                                self.terms_to_tokens[term] = set(tokens)

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback):
        pass
