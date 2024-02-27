import argparse
import json
import logging
from pathlib import Path
import pickle
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
import tqdm

from datetime import datetime, timezone

from data_sources.data_source import DataSource
from data_sources.basic_csv import BasicCSVDataSource
from inference.infer import FlatClassifier

parser = argparse.ArgumentParser(description="Benchmark an FPO classification model.")

parser.add_argument(
    "--digits",
    type=int,
    help="how many digits to classify the answer to",
    default=8,
    choices=[2, 4, 6, 8],
)

parser.add_argument(
    "--output",
    help="choose how you want the results outputted",
    type=str,
    default="text",
    choices=["text", "json"],
)

parser.add_argument(
    "--write-to-file",
    help="whether to write output to a file",
    required=False,
    default=False,
    action="store_true",
)

parser.add_argument(
    "--no-progress",
    help="don't show a progress bar",
    required=False,
    default=False,
    action="store_true",
)

parser.add_argument(
    "--colour",
    help="enable ANSI colour for the 'text' output type",
    required=False,
    default=False,
    action="store_true",
)

args = parser.parse_args()

digits = args.digits
output = args.output
no_progress = args.no_progress
colour = args.colour
write_file = args.write_to_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Everything except CPU seems to be super slow, I assume because of the time it
# takes to shuffle the data about for each inference. So we'll default to CPU.
device = "cpu"

cwd = Path(__file__).resolve().parent

target_dir = cwd / "target"
target_dir.mkdir(parents=True, exist_ok=True)

subheadings_file = target_dir / "subheadings.pkl"
if not subheadings_file.exists():
    raise FileNotFoundError(f"Could not find subheadings file: {subheadings_file}")

with open(subheadings_file, "rb") as fp:
    subheadings = pickle.load(fp)

model_file = target_dir / "model.pt"
if not model_file.exists():
    raise FileNotFoundError(f"Could not find model file: {model_file}")

classifier = FlatClassifier(model_file, subheadings, device)

data_sources: list[DataSource] = []

# Append all the benchmarking data sources
benchmarking_data_dir = cwd / "benchmarking_data"

data_sources += [
    BasicCSVDataSource(filename, 1, 0)
    for filename in benchmarking_data_dir.glob("**/*.csv")
]

benchmarking_data = {}

logger.info("Loading and processing test data")
skip_descriptions = set()

for data_source in data_sources:
    for code, descriptions in data_source.get_codes(digits).items():
        for description in descriptions:
            if description not in benchmarking_data:
                benchmarking_data[description] = code
            elif benchmarking_data[description] != code:
                logger.warning(
                    f"🚨 Duplicate description with multiple codes! {description} => {benchmarking_data[description]} & {code} - SKIPPING"
                )
                skip_descriptions.add(description)


logger.info(f"Loaded {len(benchmarking_data)} items")
logger.info(f"Skipping {len(skip_descriptions)} items with ambiguous classifications")


logger.info(
    f"Getting results for {len(benchmarking_data) - len(skip_descriptions)} items"
)

res = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "NF": 0,
    "NR": 0,
    "chapter": 0,
    "heading": 0,
    "subheading": 0,
}

num = 0
items = benchmarking_data.items()

if not no_progress:
    items = tqdm.tqdm(items)  # Use a nice progress bar if it hasn't been disabled

for description, code in items:
    if description in skip_descriptions:
        continue

    num += 1

    term_results = classifier.classify(description, 5, digits)

    found = False

    for i in range(len(term_results)):
        if code == term_results[i].code:
            found = True
            res[str(i + 1)] += 1
            break

    if not found:
        res["NF"] += 1

    if len(term_results) == 0:
        res["NR"] += 1

    if term_results and term_results[0].code[:2] == code[:2]:
        res["chapter"] += 1

    if term_results and term_results[0].code[:4] == code[:4]:
        res["heading"] += 1

    if term_results and term_results[0].code[:6] == code[:6]:
        res["subheading"] += 1

top5sum = res["1"] + res["2"] + res["3"] + res["4"] + res["5"]

columns = [
    "Total",
    "1st",
    "2nd",
    "3rd",
    "4th",
    "5th",
    "In Top 5",
    "Outside Top 5",
    "No Results",
    "Chapter",
    "Heading",
    "Subheading",
]

if output == "text":
    if colour:
        results = ColorTable(columns, theme=Themes.OCEAN)
    else:
        results = PrettyTable(columns)

    row = [
        str(num),
        str(res["1"]) + " (" + str(round(100 * res["1"] / num, 1)) + "%)",
        str(res["2"]) + " (" + str(round(100 * res["2"] / num, 1)) + "%)",
        str(res["3"]) + " (" + str(round(100 * res["3"] / num, 1)) + "%)",
        str(res["4"]) + " (" + str(round(100 * res["4"] / num, 1)) + "%)",
        str(res["5"]) + " (" + str(round(100 * res["5"] / num, 1)) + "%)",
        str(top5sum) + " (" + str(round(100 * top5sum / num, 1)) + "%)",
        str(res["NF"]) + " (" + str(round(100 * res["NF"] / num, 1)) + "%)",
        str(res["NR"]) + " (" + str(round(100 * res["NR"] / num, 1)) + "%)",
        str(res["chapter"]) + " (" + str(round(100 * res["chapter"] / num, 1)) + "%)",
        str(res["heading"])
        + (
            " (" + str(round(100 * res["heading"] / num, 1)) + "%)"
            if digits >= 4
            else "-"
        ),
        str(res["subheading"])
        + (
            " (" + str(round(100 * res["subheading"] / num, 1)) + "%)"
            if digits >= 6
            else "-"
        ),
    ]

    results.add_row(row)

    print(results)
else:  # output is json
    row = {
        "total": num,
        "result1": res["1"],
        "result1Percent": 100 * res["1"] / num,
        "result2": res["2"],
        "result2Percent": 100 * res["2"] / num,
        "result3": res["3"],
        "result3Percent": 100 * res["3"] / num,
        "result4": res["4"],
        "result4Percent": 100 * res["4"] / num,
        "result5": res["5"],
        "result5Percent": 100 * res["5"] / num,
        "inTop5": top5sum,
        "inTop5Percent": 100 * top5sum / num,
        "outsideTop5": res["NF"],
        "outsideTop5Percent": 100 * res["NF"] / num,
        "noResults": res["NR"],
        "noResultsPercent": 100 * res["NR"] / num,
        "inChapter": res["chapter"],
        "inChapterPercent": 100 * res["chapter"] / num,
        "inHeading": res["heading"],
        "inHeadingPercent": 100 * res["heading"] / num,
        "inSubheading": res["subheading"],
        "inSubheadingPercent": 100 * res["subheading"] / num,
    }

    print(json.dumps(row))

if write_file:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filetype = "json" if output == "json" else "txt"
    file = open(f"benchmark_results_{timestamp}.{filetype}", "w")

    if output == "json":
        file.write(json.dumps(row))
    else:
        file.write(results)

    file.close()
