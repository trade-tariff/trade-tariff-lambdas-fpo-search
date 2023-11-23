import json
import os
import unittest

test_fpo_client_keys = {"test_id": "test_secret"}

os.environ["FPO_CLIENT_KEYS"] = json.dumps(test_fpo_client_keys)

from handler import handle  # noqa: E402


class Test_handler_handle(unittest.TestCase):
    def test_it_should_return_unauthorised_with_no_headers(self):
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {"q": "test", "digits": "6", "limit": "5"},
            "headers": {},
        }

        result = handle(event, {})

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

        result = handle(event, {})

        self.assertEqual(401, result["statusCode"], "Expected a 401 status code")

    def test_it_should_return_unauthorised_with_invalid_secret(self):
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {"q": "test", "digits": "6", "limit": "5"},
            "headers": {"x-api-client-id": "test_id", "x-api-secret-key": "invalid"},
        }

        result = handle(event, {})

        self.assertEqual(401, result["statusCode"], "Expected a 401 status code")

    def test_it_should_handle_a_valid_get_request(self):
        event = self._create_get_event("test")

        result = handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(5, len(result_body["results"]), "Expected 5 results")

    def test_it_should_handle_a_valid_post_request(self):
        event = self._create_post_event("test")

        result = handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(5, len(result_body["results"]), "Expected 5 results")

    def test_it_should_respect_the_limit(self):
        event = self._create_post_event("test", "6", "2")

        result = handle(event, {})
        result_body = json.loads(result["body"])

        self.assertEqual(200, result["statusCode"], "Expected a 200 status code")
        self.assertEqual(2, len(result_body["results"]), "Expected 2 results")

    def test_it_should_handle_invalid_digits(self):
        event = self._create_post_event("test", "invalid")

        result = handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def test_it_should_handle_too_many_digits(self):
        event = self._create_post_event("test", "10")

        result = handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def test_it_should_handle_invalid_limit(self):
        event = self._create_post_event("test", "6", "invalid")

        result = handle(event, {})

        self.assertEqual(400, result["statusCode"], "Expected a 400 status code")

    def test_it_should_handle_too_high_limit(self):
        event = self._create_post_event("test", "6", "11")

        result = handle(event, {})

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


if __name__ == "__main__":
    unittest.main()
