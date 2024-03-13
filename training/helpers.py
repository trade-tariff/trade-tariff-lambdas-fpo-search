class Helpers:
    @staticmethod
    def unique_words(string):
        words = string.split()
        unique_words = []

        for word in words:
            if word not in unique_words:
                unique_words.append(word)

        unique_string = " ".join(unique_words)
        return unique_string
