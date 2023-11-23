from pathlib import Path
import pickle
import time
import os
import json
from inference.infer import FlatClassifier

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

fpo_client_keys = json.loads(os.environ.get("FPO_CLIENT_KEYS", "{}"))
print(f"🚀⇨ Loaded client keys: {fpo_client_keys.keys()}")


def handle(event, _context):
    if event.get("httpMethod", "GET") == "POST":
        body = json.loads(event.get("body", {}))

        description = body.get("description", "")
        digits = body.get("digits", 6)
        limit = body.get("limit", 5)
    else:
        queryParams = event.get("queryStringParameters", {})

        description = queryParams.get("q", "")
        digits = queryParams.get("digits", 6)
        limit = queryParams.get("limit", 5)

    statusCode = 200
    body = {}

    if description == "":
        statusCode = 400
        body = {"message": "No description specified"}
    elif digits not in ["6", "8"]:
        statusCode = 400
        body = {"message": "Invalid digits"}
    elif not limit.isdecimal() or int(limit) < 1 or int(limit) > 10:
        statusCode = 400
        body = {"message": "Invalid limit"}
    elif not authorised(event):
        statusCode = 401
        body = {"message": "Unauthorized"}
    else:
        results = classifier.classify(description, int(limit), int(digits))
        body = {
            "results": [
                {"code": result.code, "score": result.score * 1000}
                for result in results
            ]
        }

    return {"statusCode": statusCode, "body": json.dumps(body)}


def authorised(event):
    headers = event.get("headers", {})
    headers = {k.lower(): v for k, v in headers.items()}
    client_id = headers.get("x-api-client-id")
    api_key = headers.get("x-api-secret-key")

    if client_id is None:
        print("⚠️ No client id specified")
        return False

    if client_id not in fpo_client_keys:
        print(f"⚠️ Invalid client id '{client_id}' specified")
        return False

    expected_key = fpo_client_keys.get(client_id, "")

    return api_key == expected_key
