import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


class GPT2SpellingCorrector:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def correct(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        # Generate attention mask
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=input_ids.device
        )

        # Generate corrected spelling
        corrected_text = self.model.generate(
            input_ids,
            max_length=100,
            attention_mask=attention_mask,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(corrected_text[0], skip_special_tokens=True)


if __name__ == "__main__":
    sc = GPT2SpellingCorrector()

    text = """Correct the spelling of the following sentence:
    WHITW WINE"""

    print(f"input:{text}")

    corrected_text = sc.correct(text)
    print(f"corrected:{corrected_text}")
