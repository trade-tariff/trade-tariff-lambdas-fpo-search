from pathlib import Path
import pickle
import time
import os
import json

from inference.infer import FlatClassifier

digits = 6
limit = 5
cwd = Path(__file__).resolve().parent
target_dir = cwd / "target"
subheadings_file = target_dir / "subheadings.pkl"
with open(subheadings_file, "rb") as fp:
    subheadings = pickle.load(fp)

model_file = target_dir / "model.pt"

start = time.time()
print(f"🚀⇨ Loading static classifier from {model_file}")
model_exists = os.path.isfile(model_file)
print(f"🚀⇨ Model exists: {model_exists}")
classifier = FlatClassifier(model_file, subheadings)
print(f"🚀⇨ Static classifier loaded in {time.time() - start:.2f}s")


def handle(event, _context):
    queryParams = event.get("queryStringParameters", {})
    headers = event.get("headers", {})
    headers = {k.lower(): v for k, v in headers.items()}
    api_key = headers.get("x-api-key", "")

    q = queryParams.get("q", "")
    digits = int(queryParams.get("digits", 6))

    statusCode = 200
    body = {}

    if digits not in [6, 8]:
        statusCode = 400
        body = {"message": "Invalid digits"}
    elif api_key != os.environ.get("API_KEY"):
        statusCode = 401
        body = {"message": "Unauthorized"}
    else:
        results = classifier.classify(q, limit, digits)
        body = {
            "results": [
                {"code": result.code, "score": result.score * 1000}
                for result in results
            ]
        }

    return {"statusCode": statusCode, "body": json.dumps(body)}