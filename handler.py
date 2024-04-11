from pathlib import Path
import pickle
import time
import os

from aws_lambda.handler import LambdaHandler
from aws_lambda_powertools import Logger

from inference.infer import FlatClassifier

logger = Logger(service="fpo-commodity-code-tool")

start = time.perf_counter()
cwd = Path(__file__).resolve().parent
target_dir = cwd / "target"
subheadings_file = target_dir / "subheadings.pkl"
with open(subheadings_file, "rb") as fp:
    subheadings = pickle.load(fp)
logger.info("🚀⇨ Subheadings loaded in %.2fms", (time.perf_counter() - start) * 1000)

model_file = target_dir / "model.pt"

start = time.perf_counter()
logger.info("🚀⇨ Loading static classifier from %s", model_file)

model_exists = os.path.isfile(model_file)
logger.info("🚀⇨ Model exists: %s", model_exists)

classifier = FlatClassifier(model_file, subheadings, "cpu")
logger.info(
    "🚀⇨ Static classifier loaded in %.2fms", (time.perf_counter() - start) * 1000
)
lambda_handler = LambdaHandler(classifier, logger=logger)


@logger.inject_lambda_context
def handle(event, _context):
    return lambda_handler.handle(event, _context)
