import toml
from data_sources.vague_terms import VagueTermsCSVDataSource
from train_args import TrainScriptArgsParser
import logging
from pathlib import Path

import pickle

import torch
from data_sources.search_references import SearchReferencesDataSource
from data_sources.data_source import DataSource
from data_sources.basic_csv import BasicCSVDataSource
from training.cleaning_pipeline import (
    CleaningPipeline,
    LanguageCleaning,
    RemoveDescriptionsMatchingRegexes,
    RemoveEmptyDescription,
    RemoveShortDescription,
    RemoveSubheadingsNotMatchingRegexes,
    StripExcessWhitespace,
)

from training.prepare_data import TrainingDataLoader
from training.train_model import FlatClassifierModelTrainer
from training.create_embeddings import EmbeddingsProcessor

args = TrainScriptArgsParser()
args.print()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

cwd = Path(__file__).resolve().parent

target_dir = args.target_dir()
target_dir.mkdir(parents=True, exist_ok=True)

data_dir = args.data_dir()
data_dir.mkdir(parents=True, exist_ok=True)

args.reference_dir().mkdir(parents=True, exist_ok=True)

# First load in the training data
print("💾⇨ Loading training data")

subheadings_file = target_dir / "subheadings.pkl"

language_skips_file = args.pwd() / args.partial_non_english_terms()
language_keeps_file = args.pwd() / args.partial_english_terms()
language_keeps_exact_file = args.pwd() / args.exact_english_terms()

with open(language_skips_file, "r") as f:
    language_skips = f.read().splitlines()

with open(language_keeps_file, "r") as f:
    language_keeps = f.read().splitlines()

with open(language_keeps_exact_file, "r") as f:
    language_keeps_exact = f.read().splitlines()

basic_filters = [
    StripExcessWhitespace(),
    RemoveEmptyDescription(),
    RemoveShortDescription(min_length=4),
    RemoveSubheadingsNotMatchingRegexes(
        regexes=[
            "^\\d{" + str(args.digits()) + "}$",
        ]
    ),
]
tradestats_filters = basic_filters + [
    RemoveDescriptionsMatchingRegexes(
        regexes=[
            r"^\\d+$",  # Skip rows where description contains only numbers
            r"^[0-9-]+$",  # Skip rows where description contains only numbers and dashes
            r"^[./]+$",  # Skip rows where description consists only of a '.' or a '/'
            r"^\d+-\d+$",  # skip numbers with hyphens in between
            r"^[0-9*]+$",  # Skip rows where description contains only numbers and asterisks
            r"^[-+]?\d+(\.\d+)?$",  # skip if just decimal numbers
            r"^\d+\s+\d+$",  # Skip rows where description contains one or more digits and one or more whitespace characters (including spaces, tabs, and other Unicode spaces)
            r"^[0-9,]+$",  # Skip rows where description contains only numbers and commas
        ]
    ),
    LanguageCleaning(
        detected_languages=args.detected_languages(),
        preferred_languages=args.preferred_languages(),
        partial_skips=language_skips,
        partial_keeps=language_keeps,
        exact_keeps=language_keeps_exact,
    ),
]

basic_pipeline = CleaningPipeline(basic_filters)
tradestats_pipeline = CleaningPipeline(tradestats_filters)

data_sources: list[DataSource] = []

data_sources.append(VagueTermsCSVDataSource(args.vague_terms_data_file()))

data_sources.append(
    BasicCSVDataSource(
        args.extra_references_data_file(),
        code_col=1,
        description_col=0,
        authoritative=True,
        creates_codes=False,
    )
)

data_sources.append(SearchReferencesDataSource())

data_sources.append(
    BasicCSVDataSource(
        args.cn_data_file(),
        code_col=1,
        description_col=3,
        cleaning_pipeline=basic_pipeline,
        authoritative=True,
        creates_codes=True,
    )
)

data_sources += [
    BasicCSVDataSource(
        filename,
        cleaning_pipeline=tradestats_pipeline,
        encoding="latin_1",
    )
    for filename in Path(args.tradesets_data_dir()).glob("*.csv")
]

training_data_loader = TrainingDataLoader()

(text_values, subheadings, texts, labels) = training_data_loader.fetch_data(
    data_sources, args.digits()
)

print(f"Found {len(text_values)} unique descriptions")

print("💾⇦ Saving subheadings")
with open(subheadings_file, "wb") as fp:
    pickle.dump(subheadings, fp)

# Impose the limit if required - this will limit the number of unique descriptions
if args.limit() is not None:
    text_values = text_values[: args.limit()]

    new_texts: list[int] = []
    new_labels: list[int] = []

    for i, t in enumerate(texts):
        if t < len(text_values):
            new_texts.append(t)
            new_labels.append(labels[i])

    texts = new_texts
    labels = new_labels

# Next create the embeddings
print("Creating the embeddings")

embeddings_processor = EmbeddingsProcessor(
    transformer_model=args.transformer(),
    torch_device=args.torch_device(),
    batch_size=args.embedding_batch_size(),
)

unique_embeddings = embeddings_processor.create_embeddings(text_values)

# Now build and train the network
trainer = FlatClassifierModelTrainer(args)

# Convert the labels to a Tensor
labels = torch.tensor(labels, dtype=torch.long)

embeddings = torch.stack([unique_embeddings[idx] for idx in texts])

state_dict, input_size, hidden_size, output_size = trainer.run(
    embeddings, labels, len(subheadings)
)

print("💾⇦ Saving model")

model_file = target_dir / "model.pt"
torch.save(state_dict, model_file)

config = toml.load("search-config.toml")
config["model_input_size"] = input_size
config["model_hidden_size"] = hidden_size
config["model_output_size"] = output_size

with open("search-config.toml", "w") as f:
    toml.dump(config, f)

print("✅ Training complete. Enjoy your model!")
