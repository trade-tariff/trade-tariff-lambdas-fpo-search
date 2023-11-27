import json

from inference.infer import Classifier


class LambdaHandler:
    def __init__(self, classifier: Classifier, api_keys: dict[str, str]) -> None:
        self._classifier = classifier
        self._api_keys = api_keys

    def handle(self, event, _context):
        if event.get("httpMethod", "GET") == "POST":
            body = json.loads(event.get("body", {}))

            description = body.get("description", "")
            digits = body.get("digits", "6")
            limit = body.get("limit", "5")
        else:
            queryParams = event.get("queryStringParameters", {})

            description = queryParams.get("q", "")
            digits = queryParams.get("digits", "6")
            limit = queryParams.get("limit", "5")

        statusCode = 200
        body = {}

        if description == "":
            statusCode = 400
            body = {"message": "No description specified"}
        elif str(digits) not in ["6", "8"]:
            statusCode = 400
            body = {"message": "Invalid digits"}
        elif not str(limit).isdecimal() or int(limit) < 1 or int(limit) > 10:
            statusCode = 400
            body = {"message": "Invalid limit"}
        elif not self._authorised(event):
            statusCode = 401
            body = {"message": "Unauthorized"}
        else:
            results = self._classifier.classify(description, int(limit), int(digits))
            body = {
                "results": [
                    {"code": result.code, "score": result.score * 1000}
                    for result in results
                ]
            }

        return {"statusCode": statusCode, "body": json.dumps(body)}

    def _authorised(self, event):
        headers = event.get("headers", {})
        headers = {k.lower(): v for k, v in headers.items()}
        client_id = headers.get("x-api-client-id")
        api_key = headers.get("x-api-secret-key")

        if client_id is None:
            print("⚠️ No client id specified")
            return False

        if client_id not in self._api_keys:
            print(f"⚠️ Invalid client id '{client_id}' specified")
            return False

        expected_key = self._api_keys.get(client_id, "")

        return api_key == expected_key
