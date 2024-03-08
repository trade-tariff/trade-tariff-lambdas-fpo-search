from openai import OpenAI


class OpenAISpellingCorrector:
    def __init__(self):
        self.client = OpenAI(api_key="---REPLACE WITH YOUR OPENAI API KEY---")

        # api_key=os.environ.get("OPENAI_API_KEY")

    def correct(self, text):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful spelling corrector."},
                {"role": "user", "content": text},
            ],
        )
        return completion.choices[0].message


if __name__ == "__main__":
    sc = OpenAISpellingCorrector()

    text = """
DESIGNER PRODUCTS
DENTAL INSTRAMENTS
TURTLA SNOOD
"""

    print(sc.correct(text))
