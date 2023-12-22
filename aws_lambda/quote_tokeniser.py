import re

"""
Converts an input query with surrounding single and double quotes and produces
an array of token tuples that include token phrase and an instruction of whether
the tokens are within single or double quotes

Example:
    query = "this is a 'test query'"
    tokeniser = QuoteTokeniser()
    tokens = tokeniser.tokenise(query)
    print(tokens)
    # [('this', True), ('is', True), ('a', True), ('test query', False)]


Parameters:
    search_term (str): The search term to be tokenised.

Returns:
    list: A list of tuples containing the token and a boolean value indicating whether the token should be corrected.
"""


class QuoteTokeniser:
    PATTERN = re.compile(r'((?:"(?:\\.|[^\\"])*")|(?:\'(?:\\.|[^\\\'])*\')|\S+)')

    @staticmethod
    def tokenise(search_term):
        if search_term is None or search_term == "":
            return []

        terms = []

        for match in QuoteTokeniser.PATTERN.finditer(search_term):
            term = match.group(0)
            if term.startswith('"') and term.endswith('"'):
                terms.append((term, True))
            elif term.startswith("'") and term.endswith("'"):
                terms.append((term, True))
            else:
                terms.append((term, False))
        return terms
