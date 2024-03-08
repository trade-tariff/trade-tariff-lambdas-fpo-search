from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5SpellingCorrector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    def correct(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        sample_output = self.model.generate(
            input_ids, do_sample=True, max_length=50, top_p=0.99, num_return_sequences=1
        )
        res = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
        return res


if __name__ == "__main__":
    text = """Correct the errors in the following text:
T-SHIRT CREW NECK S/S ESSENTIA
WHITW WINE 1
GALLERY DEPT. PAINT-SPLATTER DISTRE
INDEXABLE IINSERT
JEWELLARY
GU8MMIES
TANGY APPLE FLAVAOUR LABLES
"""

    t5 = T5SpellingCorrector()

    print(t5.correct(text))
