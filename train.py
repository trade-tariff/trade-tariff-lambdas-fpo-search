from data_sources.vague_terms import VagueTermsCSVDataSource
from fpo_args_parser import FPOArgsParser
import logging
from pathlib import Path
import pickle

import torch
from data_sources.search_references import SearchReferences
from data_sources.data_source import DataSource
from data_sources.basic_csv import BasicCSVDataSource
from training.create_embeddings import EmbeddingsProcessor
from training.prepare_data import TrainingDataLoader
from training.train_model import (
    FlatClassifierModelTrainer,
    FlatClassifierModelTrainerParameters,
)

args = FPOArgsParser().parsed_args

limit = args.limit
force = args.force
batch_size = args.batch_size
embeddings_batch_size = args.embedding_batch_size
embedding_cache_checkpoint = args.embedding_cache_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

training_parameters = FlatClassifierModelTrainerParameters(
    args.learning_rate, args.max_epochs
)

device = FPOArgsParser().torch_device()
print(f"⚙️  Using device {device}")

cwd = Path(__file__).resolve().parent

target_dir = cwd / "target"
target_dir.mkdir(parents=True, exist_ok=True)

data_dir = target_dir / "training_data"
data_dir.mkdir(parents=True, exist_ok=True)

# First load in the training data
print("💾⇨ Loading training data")

search_references = SearchReferences()

text_values_file = data_dir / "text_values.pkl"
texts_file = data_dir / "texts.pkl"
labels_file = data_dir / "labels.pkl"
subheadings_file = target_dir / "subheadings.pkl"

if (
    not force
    and texts_file.exists()
    and labels_file.exists()
    and subheadings_file.exists()
):
    print("💾⇨ Text values pickle file found. Loading...")
    with open(text_values_file, "rb") as fp:
        text_values = pickle.load(fp)

    print("💾⇨ Texts pickle file found. Loading...")
    with open(texts_file, "rb") as fp:
        texts = pickle.load(fp)

    print("💾⇨ Labels pickle file found. Loading...")
    with open(labels_file, "rb") as fp:
        labels = pickle.load(fp)

    print("💾⇨ Subheadings pickle file found. Loading...")
    with open(subheadings_file, "rb") as fp:
        subheadings = pickle.load(fp)
else:
    data_sources: list[DataSource] = []

    reference_data_dir = cwd / "reference_data"

    # Vague terms data source
    vague_terms_data_file = reference_data_dir / "vague_terms.csv"

    data_sources.append(VagueTermsCSVDataSource(vague_terms_data_file))

    # Search references data source
    data_sources.append(search_references)

    # Combined Nomenclature self-explanatory data source
    cn_data_file = reference_data_dir / "CN2024_SelfText_EN_DE_FR.csv"

    data_sources.append(
        BasicCSVDataSource(
            cn_data_file,
            code_col=1,
            description_col=3,
            authoritative=True,
            creates_codes=True,
        )
    )

    # Append all the Tradesets data sources
    source_dir = cwd / "raw_source_data"

    tradesets_data_dir = source_dir / "tradesets_descriptions"

    data_sources += [
        BasicCSVDataSource(filename, encoding="latin_1")
        for filename in tradesets_data_dir.glob("*.csv")
    ]

    training_data_loader = TrainingDataLoader()

    (text_values, subheadings, texts, labels) = training_data_loader.fetch_data(
        data_sources, 8
    )

    print("💾⇦ Saving text values")
    with open(text_values_file, "wb") as fp:
        pickle.dump(text_values, fp)

    print("💾⇦ Saving subheadings")
    with open(subheadings_file, "wb") as fp:
        pickle.dump(subheadings, fp)

    print("💾⇦ Saving text indexes")
    with open(texts_file, "wb") as fp:
        pickle.dump(texts, fp)

    print("💾⇦ Saving label indexes")
    with open(labels_file, "wb") as fp:
        pickle.dump(labels, fp)

# Impose the limit if required - this will limit the number of unique descriptions
if limit is not None:
    text_values = text_values[:limit]

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
    data_dir,
    torch_device=device,
    batch_size=embeddings_batch_size,
    cache_checkpoint=embedding_cache_checkpoint,
)

unique_embeddings = embeddings_processor.create_embeddings(text_values)

# Now build and train the network
trainer = FlatClassifierModelTrainer(
    training_parameters, device=device, batch_size=batch_size
)

# Convert the labels to a Tensor
labels = torch.tensor(labels, dtype=torch.long)

embeddings = torch.stack([unique_embeddings[idx] for idx in texts])

model = trainer.run(embeddings, labels, len(subheadings))

print("💾⇦ Saving model")

model_file = target_dir / "model.pt"
torch.save(model, model_file)

print("✅ Training complete. Enjoy your model!")
