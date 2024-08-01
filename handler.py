from pathlib import Path
import pickle
import os
import sentry_sdk

from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration
from aws_lambda.handler import LambdaHandler
from aws_lambda_powertools import Logger

from inference.infer import FlatClassifier
from utils.timer import CodeTimerFactory

logger = Logger(service="fpo-commodity-code-tool")

code_timer_factory = CodeTimerFactory(logger=logger)

with code_timer_factory.time_code("Load subheadings"):
    cwd = Path(__file__).resolve().parent
    target_dir = cwd / "target"
    subheadings_file = target_dir / "subheadings.pkl"
    with open(subheadings_file, "rb") as fp:
        subheadings = pickle.load(fp)

with code_timer_factory.time_code("Loading static classifier"):
    logger.info("🚀⇨ Loading static classifier")
    classifier = FlatClassifier(subheadings, device="cpu", logger=logger)

lambda_handler = LambdaHandler(classifier, logger=logger)


def strip_sensitive_headers(event, _hint):
    if "request" in event:
        request = event["request"]
        headers = request.get("headers", {})
        multi_value_headers = request.get("multiValueHeaders", {})

        headers = {k.lower(): v for k, v in headers.items()}
        multi_value_headers = {k.lower(): v for k, v in multi_value_headers.items()}

        headers.pop("x-api-key", None)
        multi_value_headers.pop("x-api-key", None)

        event["request"]["headers"] = headers
        event["request"]["multiValueHeaders"] = multi_value_headers

    return event


with code_timer_factory.time_code("Initialising Sentry"):
    sentry_sdk.init(
        os.getenv("SENTRY_DSN", ""),
        integrations=[AwsLambdaIntegration(timeout_warning=True)],
        environment=os.getenv("SENTRY_ENVIRONMENT", ""),
        before_send=strip_sensitive_headers,
    )


@logger.inject_lambda_context
def handle(event, context):
    return lambda_handler.handle(event, context)
