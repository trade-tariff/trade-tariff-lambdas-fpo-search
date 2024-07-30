import json
from typing import Union

import aws_lambda_powertools

from inference.infer import Classifier
import logging
import time

with open("REVISION", "r") as f:
    REVISION = f.read().strip()

with open("MODEL_VERSION", "r") as f:
    MODEL_VERSION = f.read().strip()


def log_handler(func):
    def wrapper(self, event, _context):
        start = time.perf_counter()
        result = func(self, event, _context)
        lapsed = (time.perf_counter() - start) * 1000

        self._logger.info(
            "Handler %s completed in %.2fms",
            func.__name__,
            lapsed,
            extra={
                "http_method": event.get("httpMethod"),
                "path": event.get("path"),
                "status_code": result["statusCode"],
                "time_ms": lapsed,
            },
        )
        return result

    return wrapper


class LambdaHandler:
    def __init__(
        self,
        classifier: Classifier,
        logger: aws_lambda_powertools.Logger | logging.Logger = logging.getLogger(
            "handler"
        ),
    ) -> None:
        self._classifier = classifier
        self._logger = logger

    def handle(self, event, _context):
        if isinstance(self._logger, aws_lambda_powertools.Logger):
            api_key_id = (
                event.get("requestContext", {}).get("identity", {}).get("apiKeyId", "")
            )
            request_id = event.get("requestContext", {}).get("requestId", "")
            user_agent = event.get("headers", {}).get("User-Agent", "")
            self._logger.append_keys(
                api_key_id=api_key_id, request_id=request_id, user_agent=user_agent
            )

        http_method = event.get("httpMethod", "GET")
        path = event.get("path", "default")

        method_name = (
            f'handle_{path.strip("/").replace("-", "_")}_{http_method.lower()}'
        )
        handler = getattr(self, method_name, self.handle_default)

        return handler(event, _context)

    @log_handler
    def handle_fpo_code_search_post(self, event, _context):
        try:
            body = json.loads(event.get("body", {}))
        except json.JSONDecodeError as e:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"message": "Invalid JSON in request body", "detail": str(e)}
                ),
                "headers": self._headers(event),
            }

        description = body.get("description", "")
        digits = body.get("digits", "6")
        limit = body.get("limit", "5")

        response = self._handle_classification(description, digits, limit)
        response["headers"] = self._headers(event)

        return response

    def handle_healthcheck_get(self, event, _context):
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "git_sha1": REVISION,
                    "model_version": MODEL_VERSION,
                    "healthy": True,
                }
            ),
        }

    @log_handler
    def handle_default(self, event, _context):
        return {
            "statusCode": 404,
            "body": json.dumps({"message": "Not Found"}),
            "headers": self._headers(event),
        }

    def _handle_classification(
        self, description: str, digits: Union[str, int], limit: Union[str, int]
    ):
        valid = self._validate(description, digits, limit)

        if not valid[0]:
            return {
                "statusCode": 400,
                "body": json.dumps({"message": f"Invalid value for {valid[1]}"}),
            }

        results = self._classifier.classify(description, int(limit), int(digits))
        results = [
            {"code": result.code, "score": result.score * 1000} for result in results
        ]

        self._logger.info(
            "Inference result",
            extra={
                "request_description": description,
                "request_digits": int(digits),
                "request_limit": int(limit),
                "result_count": len(results),
                "results": results,
            },
        )

        return {"statusCode": 200, "body": json.dumps({"results": results})}

    def _validate(
        self, description: str, digits: Union[str, int], limit: Union[str, int]
    ):
        if not description:
            return (False, "description")

        if str(digits) not in ["6", "8"]:
            return (False, "digits")

        if not str(limit).isdecimal() or int(limit) < 1 or int(limit) > 10:
            return (False, "limit")

        return (True, "")

    def _headers(self, event):
        return {
            "Content-Type": "application/json",
            "X-Request-Id": event.get("requestContext", {}).get("requestId", ""),
        }
