import json

import aws_lambda_powertools

from inference.infer import Classifier
import logging
import time


class LambdaHandler:
    def __init__(
        self,
        classifier: Classifier,
        api_keys: dict[str, str],
        logger: aws_lambda_powertools.Logger | logging.Logger = logging.getLogger(),
    ) -> None:
        self._classifier = classifier
        self._api_keys = api_keys
        self._logger = logger

    def handle(self, event, _context):
        client_id = self._authenticate(event)

        if client_id is None:
            return {
                "statusCode": 401,
                "body": json.dumps({"message": "Unauthorized"}),
            }

        if event.get("httpMethod", "GET") == "POST":
            try:
                body = json.loads(event.get("body", {}))
            except json.JSONDecodeError as e:
                return {
                    "statusCode": 400,
                    "body": json.dumps(
                        {"message": "Invalid JSON in request body", "detail": str(e)}
                    ),
                }

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
        else:
            start = time.perf_counter()
            raw_results = self._classifier.classify(
                description, int(limit), int(digits)
            )
            results = [
                {"code": result.code, "score": result.score * 1000}
                for result in raw_results
            ]
            body = {"results": results}
            lapsed = (time.perf_counter() - start) * 1000

            self._logger.info(
                "Results generated in %.2fms",
                lapsed,
                extra={
                    "client_id": client_id,
                    "request_description": description,
                    "request_digits": int(digits),
                    "request_limit": int(limit),
                    "result_time_ms": lapsed,
                    "result_count": len(results),
                    "results": results,
                },
            )

        return {"statusCode": statusCode, "body": json.dumps(body)}

    def _authenticate(self, event) -> str | None:
        headers = event.get("headers", {})
        headers = {k.lower(): v for k, v in headers.items()}
        client_id = headers.get("x-api-client-id")
        api_key = headers.get("x-api-secret-key")

        if client_id is None:
            self._logger.info("No client id specified")
            return None

        if client_id not in self._api_keys:
            self._logger.info("Invalid client id '%s' specified", client_id)
            return None

        expected_key = self._api_keys.get(client_id, "")

        if api_key == expected_key:
            self._logger.debug("Authenticated client '%s'", client_id)
            return client_id

        self._logger.info("Invalid secret key for client id '%s' specified", client_id)
        return None
