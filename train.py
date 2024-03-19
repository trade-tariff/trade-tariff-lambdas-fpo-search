from fpo_args_parser import FPOArgsParser
import logging
from pathlib import Path
import pickle
import numpy as np
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
import sklearn.model_selection as sk


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
datafile_file = data_dir / "datafile.pkl"
subheadings_file = target_dir / "subheadings.pkl"

if (
    not force
    and texts_file.exists()
    and labels_file.exists()
    and datafile_file.exists()
    and subheadings_file.exists()
):
    print("💾⇨ Texts pickle file found. Loading...")
    with open(texts_file, "rb") as fp:
        texts = pickle.load(fp)

    print("💾⇨ Labels pickle file found. Loading...")
    with open(labels_file, "rb") as fp:
        labels = pickle.load(fp)
        
    print("💾⇨ Datafile pickle file found. Loading...")
    with open(datafile_file, "rb") as fp:
        datafile = pickle.load(fp)

    print("💾⇨ Subheadings pickle file found. Loading...")
    with open(subheadings_file, "rb") as fp:
        subheadings = pickle.load(fp)
else:
    data_sources: list[DataSource] = []

    source_dir = cwd / "raw_source_data"

    # Trade tariff descriptions data source
    trade_tariff_data_file = source_dir / "commodities_uk.csv"

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

    (texts, labels, datafile, subheadings) = training_data_loader.fetch_data(data_sources, 8)

    if limit is not None:
        texts = texts[:limit]
        labels = labels[:limit]
        datafile = datafile[:limit]

    ###Remove incorrect codes and corresponding texts and labels (these will be extra ones added from outside of commodities file)
    max_label1=max(loc for loc, val in enumerate(datafile) if val == 'commodities_uk.csv')
    print(f"Up to index {max_label1} is from commodities")
    max_label = max(labels[0: max_label1+1])


    filter_items = FilterItems(texts, labels, datafile, subheadings, max_label)
    texts, labels, datafile = filter_items.filter_items() 
    subheadings = filter_items.filter_items2()

    print(f"length of texts, labels is {len(texts)}, length of subheadings is {len(subheadings)}, max subheading is {max(subheadings)}, max labels is {max(labels)}")


    ##Replace vague terms labels with new label (must be run AFTER filter out incorrect codes):
    df=pd.read_csv("Vague Terms Dictionary.csv")
    df2=df['Unacceptable_words'].str.lower() 
    vagueterms=df2.to_list()

    mapper = TextLabelMapper(texts, labels, vagueterms, subheadings)
    mapper.update_labels()
    labels = mapper.get_updated_labels() #new updated labels list
    mapper.update_subheadings()

    print({texts[{max_label1+1}]},{labels[{max_label1+1}]})

    print("💾⇦ Saving texts")
    with open(texts_file, "wb") as fp:
        pickle.dump(texts, fp)

    print(len(texts))

    print("💾⇦ Saving labels")
    with open(labels_file, "wb") as fp:
        pickle.dump(labels, fp)

    print("💾⇦ Saving datafile")
    print(len(datafile))
    with open(datafile_file, "wb") as fp:
        pickle.dump(datafile, fp)

    print(len(datafile))

    print("💾⇦ Saving subheadings")
    with open(subheadings_file, "wb") as fp:
        pickle.dump(subheadings, fp)

    print(len(subheadings))


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

print(len(embeddings))


# Now build and train the network
trainer = FlatClassifierModelTrainer(training_parameters, device)

# Convert the labels to a Tensor
labels = torch.tensor(labels, dtype=torch.long)


####Keep commodities separated first###
train_texts1=embeddings[0:max_label1+1] #first 16568 entries were from commodities file (on 8 digit version), i.e. to index 16567
temp_texts2=embeddings[max_label1+1:] #all except the first 16568 entries

train_labels1=labels[0:max_label1+1] #first 16568 entries are from commodities file (on 8 digit version), i.e. to index 16567
temp_labels2=labels[max_label1+1:]

##Take 20% random sample from rest of data, seed set at 0
from sklearn.model_selection import train_test_split
train_texts2, X_test, train_labels2, y_test = train_test_split(temp_texts2,temp_labels2, test_size=0.2, random_state=0)


X_train=torch.cat([train_texts1, train_texts2], dim=0)
y_train=torch.cat([train_labels1, train_labels2], dim=0)

print(f"Length of X_train is {len(X_train)}, length of y_train is {len(y_train)}")



model = trainer.run(X_train, X_test, y_train, y_test, len(subheadings))

print("💾⇦ Saving model")

model_file = target_dir / "model.pt"
torch.save(model, model_file)

print("✅ Training complete. Enjoy your model!")
