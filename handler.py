from pathlib import Path
import pickle
import time
import os
import json
from inference.infer import FlatClassifier

from aws_lambda.handler import LambdaHandler

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

lambda_handler = LambdaHandler(classifier, fpo_client_keys)


def handle(event, _context):
    return lambda_handler.handle(event, _context)
