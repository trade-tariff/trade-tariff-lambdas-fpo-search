class SynonymExpander:
    def __init__(self, terms_to_tokens, query=""):
        self.terms_to_tokens = terms_to_tokens
        self.query = query

    def expand(self, query):
        phrases = sorted(self.__find_matching_phrases(query))
        words = sorted(self.__find_matching_words(query))
        expanded = self.__substitute(query, phrases + words)

        return expanded

    def __substitute(self, query, words_and_phrases):
        for word_or_phrase in words_and_phrases:
            all_tokens = sorted(self.terms_to_tokens[word_or_phrase])
            all_tokens = " ".join(all_tokens)

            if word_or_phrase in query:
                query = query.replace(word_or_phrase, all_tokens)
            else:
                query = query + " " + all_tokens

        return query

    def __find_matching_phrases(self, query):
        words = query.split()

        matching_phrases = []

        for i, _word in enumerate(words[:-1]):
            phrase = " ".join(words[i : i + 2])
            if phrase in self.terms_to_tokens:
                matching_phrases.append(phrase)

        # Filter out phrases that have no corresponding tokens
        matching_phrases = [p for p in matching_phrases if self.terms_to_tokens[p]]

        return matching_phrases

    def __find_matching_words(self, query):
        words = query.split()

        matching_words = [w for w in words if w in self.terms_to_tokens]

        # Filter out words that have no corresponding tokens
        matching_words = [w for w in matching_words if self.terms_to_tokens[w]]

        return matching_words

    def __enter__(self):
        return self.expand(self.query)

    def __exit__(self, exc_type, exc_value, traceback):
        pass
