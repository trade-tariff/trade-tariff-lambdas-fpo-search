import json
import logging
import unittest
from aws_lambda.handler import LambdaHandler

from inference.infer import ClassificationResult, Classifier

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


class MockClassifier(Classifier):
    def classify(
        self, search_text: str, limit: int = 5, digits: int = 6
    ) -> list[ClassificationResult]:
        return [
            ClassificationResult(str(i).zfill(digits), (1000 - i) / 1000)
            for i in range(0, limit)
        ]


classifier = MockClassifier()

handler = LambdaHandler(classifier)


class Test_handler_handle(unittest.TestCase):
    def test_it_should_handle_a_valid_post_request(self):
        event = self._create_post_event("foo", "6", "5")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(5, len(result_body["results"]), "Expected 5 results")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_search_references(self):
        event = self._create_post_event("ricotta", "6", "5")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(
            {"results": [{"code": "040610", "score": 1000.0}]},
            result_body,
            "Expected 1 result",
        )
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_vague_terms(self):
        event = self._create_post_event("Bits", "6", "5")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(
            {"results": [{"code": "vvvvvv", "score": 0.00}]},
            result_body,
            "Expected 1 result",
        )
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_a_valid_post_request_with_ints(self):
        event = self._create_post_event_ints("foo", 6, 5)

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(5, len(result_body["results"]), "Expected 5 results")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_a_valid_post_request_default_args(self):
        event = self._create_post_event_default("foo")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(
            6,
            len(result_body["results"][0]["code"]),
            "Expected default results to have 6 digits",
        )
        self.assertEqual(5, len(result_body["results"]), "Expected 5 default results")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_respect_the_limit(self):
        event = self._create_post_event("foo", "6", "2")

        result = handler.handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(2, len(result_body["results"]), "Expected 2 results")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_invalid_digits(self):
        event = self._create_post_event("foo", "invalid")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def test_it_should_handle_too_many_digits(self):
        event = self._create_post_event("foo", "10")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_invalid_limit(self):
        event = self._create_post_event("foo", "6", "invalid")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_too_high_limit(self):
        event = self._create_post_event("foo", "6", "11")

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_invalid_json(self):
        event = self._create_post_event_default("foo")
        event["body"] = "invalid json"

        result = handler.handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_unknown_path(self):
        event = self._create_unhandled_event()

        result = handler.handle(event, {})
        self.assertEqual(404, result["statusCode"], "Expected a 404 status code")
        self.assertEqual(
            "6b85ab53-2b60-4178-81ce-342acdec65a2",
            result["headers"]["X-Request-Id"],
            "Expected a request id",
        )

    def test_it_should_handle_healthcheck(self):
        event = self._create_healthcheck_event()

        result = handler.handle(event, {})
        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(
            "development",
            json.loads(result["body"])["git_sha1"],
            "Expected a 200 status code",
        )

    def _create_post_event(self, description: str, digits: str = "6", limit: str = "5"):
        return {
            "path": "/fpo-code-search",
            "httpMethod": "POST",
            "body": json.dumps(
                {"description": description, "digits": digits, "limit": limit}
            ),
            "requestContext": {"requestId": "6b85ab53-2b60-4178-81ce-342acdec65a2"},
        }

    def _create_post_event_ints(
        self, description: str, digits: int = 6, limit: int = 5
    ):
        return {
            "path": "/fpo-code-search",
            "httpMethod": "POST",
            "body": json.dumps(
                {"description": description, "digits": digits, "limit": limit}
            ),
            "requestContext": {"requestId": "6b85ab53-2b60-4178-81ce-342acdec65a2"},
        }

    def _create_post_event_default(self, description: str):
        return {
            "path": "/fpo-code-search",
            "httpMethod": "POST",
            "body": json.dumps({"description": description}),
            "requestContext": {"requestId": "6b85ab53-2b60-4178-81ce-342acdec65a2"},
        }

    def _create_unhandled_event(self):
        return {
            "path": "/unknown",
            "httpMethod": "GET",
            "requestContext": {"requestId": "6b85ab53-2b60-4178-81ce-342acdec65a2"},
        }

    def _create_healthcheck_event(self):
        return {
            "path": "/healthcheck",
            "httpMethod": "GET",
            "requestContext": {"requestId": "6b85ab53-2b60-4178-81ce-342acdec65a2"},
        }


if __name__ == "__main__":
    unittest.main()
