import argparse
import logging
import pickle
from pathlib import Path

from inference.infer import FlatClassifier

device = "cpu"

cwd = Path(__file__).resolve().parent

target_dir = cwd / "target"

parser = argparse.ArgumentParser(description="Query an FPO classification model.")
parser.add_argument("--query", help="the query string", action="append", required=True)
parser.add_argument(
    "--limit",
    type=int,
    help="limit the number of responses",
    default=5,
)

parser.add_argument(
    "--digits",
    type=int,
    help="how many digits to classify the answer to",
    default=6,
    choices=[2, 4, 6, 8],
)

args = parser.parse_args()

query = args.query
limit = args.limit
digits = args.digits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer")

subheadings_file = target_dir / "subheadings.pkl"
if not subheadings_file.exists():
    raise FileNotFoundError(f"Could not find subheadings file: {subheadings_file}")

with open(subheadings_file, "rb") as fp:
    subheadings = pickle.load(fp)

classifier = FlatClassifier(subheadings, device)

for query in args.query:
    query, expected = query.split(":") if ":" in query else (query, None)
    result = classifier.classify(search_text=query, limit=limit, digits=digits)

    if expected:
        match = any(expected in r.code for r in result)
        print(
            f"Query: {query} -> Expected: {expected} -> Match: {match} -> Result: {result}"
        )
    else:
        print(f"Query: {query} -> Result: {result}")
