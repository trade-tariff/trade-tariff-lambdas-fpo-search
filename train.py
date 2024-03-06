import argparse
import logging
from pathlib import Path
import pickle
import pandas as pd

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
from training.clean_data import (FilterItems,
                                 TextLabelMapper,
)

parser = argparse.ArgumentParser(description="Train an FPO classification model.")
parser.add_argument(
    "--digits", type=int, help="how many digits to train the model to", default=8
)
parser.add_argument(
    "--limit",
    type=int,
    help="limit the training data to this many entries to speed up development testing",
    required=False,
)
parser.add_argument(
    "--force",
    help="force the regeneration of source data and embeddings",
    required=False,
    default=False,
    action="store_true",
)
parser.add_argument(
    "--learning-rate",
    dest="learning_rate",
    type=float,
    help="the learning rate to train the network with",
    default=0.001,
)
parser.add_argument(
    "--max-epochs",
    dest="max_epochs",
    type=int,
    help="the maximum number of epochs to train the network for",
    default=3,
)
parser.add_argument(
    "--device",
    type=str,
    help="the torch device to use for training. 'auto' will try to select the best device available.",
    choices=["auto", "cpu", "mps", "cuda"],
    default="auto",
)

args = parser.parse_args()

limit = args.limit
force = args.force

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

training_parameters = FlatClassifierModelTrainerParameters(
    args.learning_rate, args.max_epochs
)

device = args.device

if device == "auto":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built()
        else "cpu"
    )

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
        BasicCSVDataSource(filename)
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


###Remove incorrect codes and corresponding texts and labels (these will be extra ones added from outside of commodities file)
max_label1=max(loc for loc, val in enumerate(datafile) if val == 'source_data/commodities.csv')
print(f"Up to index {max_label1} is from commodities")
max_label = max(labels[0: max_label1+1])


filter_items = FilterItems(texts, labels, subheadings, max_label)
texts, labels = filter_items.filter_items() 
subheadings = filter_items.filter_items2() 

print(f"length of texts, labels is {len(texts)}, length of subheadings is {len(subheadings)}, max subheading is {max(subheadings)}, max labels is {max(labels)}")


##Replace vague terms labels with 9856 label (must be run AFTER filter out incorrect codes):
df=pd.read_csv("Vague Terms Dictionary.csv")
df2=df['Unacceptable_words'].str.lower() 
vagueterms=df2.to_list()

mapper = TextLabelMapper(texts, labels, vagueterms, subheadings)
mapper.update_labels()
labels = mapper.get_updated_labels() #new updated labels list
mapper.update_subheadings()


##Save updated subheadings
print("💾⇦ Saving updated subheadings")
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
