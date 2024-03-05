from fpo_args_parser import FPOArgsParser
import logging
from pathlib import Path
import pickle

import torch
from data_sources.data_source import DataSource
from data_sources.trade_tariff import TradeTariffDataSource
from data_sources.basic_csv import BasicCSVDataSource
from training.create_embeddings import create_embeddings
from training.prepare_data import TrainingDataLoader
from training.train_model import (
    FlatClassifierModelTrainer,
    FlatClassifierModelTrainerParameters,
)

args = FPOArgsParser().parsed_args

limit = args.limit
force = args.force

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

texts_file = data_dir / "texts.pkl"
labels_file = data_dir / "labels.pkl"
subheadings_file = target_dir / "subheadings.pkl"

if (
    not force
    and texts_file.exists()
    and labels_file.exists()
    and subheadings_file.exists()
):
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

    source_dir = cwd / "raw_source_data"

    # Trade tariff descriptions data source
    trade_tariff_data_file = source_dir / "commodities.csv"

    data_sources.append(TradeTariffDataSource(trade_tariff_data_file))

    # Append all the Tradesets data sources
    tradesets_data_dir = source_dir / "tradesets_descriptions"

    data_sources += [
        BasicCSVDataSource(filename 
                           #,encoding="latin_1"
                          )
        for filename in tradesets_data_dir.glob("*.csv")
    ]

    training_data_loader = TrainingDataLoader()

    (texts, labels, subheadings) = training_data_loader.fetch_data(data_sources, 8)

    if limit is not None:
        texts = texts[:limit]
        labels = labels[:limit]

    print("💾⇦ Saving texts")
    with open(texts_file, "wb") as fp:
        pickle.dump(texts, fp)

    print("💾⇦ Saving labels")
    with open(labels_file, "wb") as fp:
        pickle.dump(labels, fp)

    print("💾⇦ Saving subheadings")
    with open(subheadings_file, "wb") as fp:
        pickle.dump(subheadings, fp)


# Next create the embeddings
print("Creating the embeddings")

embeddings_file = data_dir / "embeddings.pkl"

if not force and embeddings_file.exists():
    print("💾⇨ Embeddings pickle file found. Loading...")
    with open(embeddings_file, "rb") as fp:
        embeddings = pickle.load(fp)
else:
    embeddings = create_embeddings(texts, device)

    print("💾⇦ Saving embeddings")
    with open(embeddings_file, "wb") as fp:
        pickle.dump(embeddings, fp)


# Now build and train the network
trainer = FlatClassifierModelTrainer(training_parameters, device)

# Convert the labels to a Tensor
labels = torch.tensor(labels, dtype=torch.long)

model = trainer.run(embeddings, labels, len(subheadings))

print("💾⇦ Saving model")

model_file = target_dir / "model.pt"
torch.save(model, model_file)

print("✅ Training complete. Enjoy your model!")
