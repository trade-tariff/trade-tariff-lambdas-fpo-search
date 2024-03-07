class SynonymExpander:
    """
    A class to expand synonyms in a query string based on a mapping of terms to tokens.
    """

    def __init__(self, terms_to_tokens):
        """
        Initializes the SynonymExpander with a dictionary mapping terms to their tokens.
        """
        self.terms_to_tokens = terms_to_tokens

    def expand(self, query):
        """
        Expands the given query by replacing words or phrases found in the terms_to_tokens mapping
        with their tokens.
        """
        phrases = sorted(self._find_matching_phrases(query), key=len, reverse=True)
        words = sorted(self._find_matching_words(query), key=len, reverse=True)
        expanded = self._unique_words(self._substitute(query, phrases + words))

        return expanded

    def _substitute(self, query, words_and_phrases):
        """
        Replaces words or phrases in the query with their corresponding tokens.
        """
        # import pdb; pdb.set_trace()
        for word_or_phrase in words_and_phrases:
            if word_or_phrase in query:
                all_tokens = " ".join(sorted(self.terms_to_tokens[word_or_phrase]))
                if query in all_tokens:
                    all_tokens = all_tokens.replace(" " + query, "")
                    query = query + " " + all_tokens
                else:
                    query = query.replace(word_or_phrase, all_tokens)
        return query

    def _find_matching_phrases(self, query):
        """
        Finds phrases in the query that are keys in the terms_to_tokens mapping.
        """
        words = query.split()
        matching_phrases = [
            " ".join(words[i : i + 2])
            for i in range(len(words) - 1)
            if " ".join(words[i : i + 2]) in self.terms_to_tokens
        ]
        return matching_phrases

    def _find_matching_words(self, query):
        """
        Finds words in the query that are keys in the terms_to_tokens mapping.
        """
        return [w for w in query.split() if w in self.terms_to_tokens]

    def _unique_words(self, string):
        words = string.split()
        unique_words = []

        for word in words:
            if word not in unique_words:
                unique_words.append(word)

        unique_string = " ".join(unique_words)

        return unique_string
