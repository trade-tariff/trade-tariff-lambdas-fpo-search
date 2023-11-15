import argparse
from pathlib import Path
import pickle

from inference.infer import FlatClassifier

device = "cpu"

cwd = Path(__file__).resolve().parent

target_dir = cwd / "target"

parser = argparse.ArgumentParser(description="Query an FPO classification model.")
parser.add_argument("query", help="the query string")
parser.add_argument(
    "--limit",
    type=int,
    help="limit the number of responses",
    default=5,
)

args = parser.parse_args()

query = args.query

subheadings_file = target_dir / "subheadings.pkl"
if not subheadings_file.exists():
    raise FileNotFoundError(f"Could not find subheadings file: {subheadings_file}")

with open(subheadings_file, "rb") as fp:
    subheadings = pickle.load(fp)

model_file = target_dir / "model.pt"
if not model_file.exists():
    raise FileNotFoundError(f"Could not find model file: {model_file}")

classifier = FlatClassifier(model_file, subheadings, device)

print(classifier.classify(query))
