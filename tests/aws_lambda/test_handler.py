import json
import logging
import unittest
from aws_lambda.handler import LambdaHandler

from inference.infer import ClassificationResult, Classifier

test_fpo_client_keys = {"test_id": "test_secret"}

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class MockClassifier(Classifier):
    def classify(
        self, search_text: str, limit: int = 5, digits: int = 6
    ) -> list[ClassificationResult]:
        return [
            ClassificationResult(str(i).zfill(digits), (1000 - i) / 1000)
            for i in range(0, limit)
        ]


classifier = MockClassifier()

handler = LambdaHandler(classifier, test_fpo_client_keys)


class Test_handler_handle(unittest.TestCase):
    def test_it_should_handle_a_valid_get_request(self):
        event = self._create_get_event("test", "8", "5")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])
        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(5, len(result_body["results"]), "Expected 5 results")

    def test_it_should_handle_a_valid_get_request_default_args(self):
        event = self._create_get_event_default("test")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])
        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(
            6,
            len(result_body["results"][0]["code"]),
            "Expected default results to have 6 digits",
        )
        self.assertEqual(5, len(result_body["results"]), "Expected 5 default results")

    def test_it_should_handle_a_valid_post_request(self):
        event = self._create_post_event("test", "6", "5")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(5, len(result_body["results"]), "Expected 5 results")

    def test_it_should_handle_a_valid_post_request_with_ints(self):
        event = self._create_post_event_ints("test", 6, 5)

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(5, len(result_body["results"]), "Expected 5 results")

    def test_it_should_handle_a_valid_post_request_default_args(self):
        event = self._create_post_event_default("test")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(
            6,
            len(result_body["results"][0]["code"]),
            "Expected default results to have 6 digits",
        )
        self.assertEqual(5, len(result_body["results"]), "Expected 5 default results")

    def test_it_should_return_unauthorised_with_no_headers(self):
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {"q": "test", "digits": "6", "limit": "5"},
            "headers": {},
        }

        result = handler.handle(event, {})

        self.assertEqual(401, result["statusCode"], "Expected a 401 status code")

    def test_it_should_return_unauthorised_with_invalid_client_id(self):
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {"q": "test", "digits": "6", "limit": "5"},
            "headers": {
                "x-api-client-id": "invalid",
                "x-api-secret-key": "test_secret",
            },
        }

        result = handler.handle(event, {})

        self.assertEqual(401, result["statusCode"], "Expected a 401 status code")

    def test_it_should_return_unauthorised_with_invalid_secret(self):
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {"q": "test", "digits": "6", "limit": "5"},
            "headers": {"x-api-client-id": "test_id", "x-api-secret-key": "invalid"},
        }

        result = handler.handle(event, {})

        self.assertEqual(401, result["statusCode"], "Expected a 401 status code")

    def test_it_should_respect_the_limit(self):
        event = self._create_post_event("test", "6", "2")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(2, len(result_body["results"]), "Expected 2 results")

    def test_it_should_handle_invalid_digits(self):
        event = self._create_post_event("test", "invalid")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def test_it_should_handle_too_many_digits(self):
        event = self._create_post_event("test", "10")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def test_it_should_handle_invalid_limit(self):
        event = self._create_post_event("test", "6", "invalid")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def test_it_should_handle_too_high_limit(self):
        event = self._create_post_event("test", "6", "11")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def _create_get_event(self, description: str, digits: str = "6", limit: str = "5"):
        return {
            "httpMethod": "GET",
            "queryStringParameters": {
                "q": description,
                "digits": digits,
                "limit": limit,
            },
            "headers": {
                "x-api-client-id": "test_id",
                "x-api-secret-key": "test_secret",
            },
        }

    def _create_get_event_default(self, description: str):
        return {
            "httpMethod": "GET",
            "queryStringParameters": {
                "q": description,
            },
            "headers": {
                "x-api-client-id": "test_id",
                "x-api-secret-key": "test_secret",
            },
        }

    def _create_post_event(self, description: str, digits: str = "6", limit: str = "5"):
        return {
            "httpMethod": "POST",
            "body": json.dumps(
                {"description": description, "digits": digits, "limit": limit}
            ),
            "headers": {
                "x-api-client-id": "test_id",
                "x-api-secret-key": "test_secret",
            },
        }

    def _create_post_event_ints(
        self, description: str, digits: int = 6, limit: int = 5
    ):
        return {
            "httpMethod": "POST",
            "body": json.dumps(
                {"description": description, "digits": digits, "limit": limit}
            ),
            "headers": {
                "x-api-client-id": "test_id",
                "x-api-secret-key": "test_secret",
            },
        }

    def _create_post_event_default(self, description: str):
        return {
            "httpMethod": "POST",
            "body": json.dumps({"description": description}),
            "headers": {
                "x-api-client-id": "test_id",
                "x-api-secret-key": "test_secret",
            },
        }


if __name__ == "__main__":
    unittest.main()
