from fpo_args_parser import FPOArgsParser
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re

import torch
#from training.create_embeddings import create_embeddings
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
labels_section_file = data_dir / "labels_section.pkl"
#datafile_file = data_dir / "datafile.pkl"
lang_file = data_dir / "lang.pkl"
subheadings_file = target_dir / "subheadings.pkl"

if (
    not force
    and texts_file.exists()
    and labels_file.exists()
    and labels_section_file.exists()
    #and datafile_file.exists()
    and subheadings_file.exists()
):
    print("💾⇨ Texts pickle file found. Loading...")
    with open(texts_file, "rb") as fp:
        texts = pickle.load(fp)

    print("💾⇨ Labels pickle file found. Loading...")
    with open(labels_file, "rb") as fp:
        labels = pickle.load(fp)
    
    print("💾⇨ Labels_section pickle file found. Loading...")
    with open(labels_section_file, "rb") as fp:
        labels_section = pickle.load(fp)

    print("💾⇨ Datafile pickle file found. Loading...")
    with open(datafile_file, "rb") as fp:
        datafile = pickle.load(fp)

    print("💾⇨ Subheadings pickle file found. Loading...")
    with open(subheadings_file, "rb") as fp:
        subheadings = pickle.load(fp)


print(f"length of texts is {len(texts)}, length of labels is {len(labels)}, length of labels_section is {len(labels_section)}, max labels is {max(labels)}")

print(f"texts[0]: {texts[0]}")

print(f"labels type: {type(labels)}")
print(f"labels values: {labels[0:3]}")


# Next open the embeddings
print("Creating the embeddings")

print(f"len texts: {len(texts)}")

embeddings_file = data_dir / "embeddings.pkl"

if not force and embeddings_file.exists():
    print("💾⇨ Embeddings pickle file found. Loading...")
    with open(embeddings_file, "rb") as fp:
        embeddings = pickle.load(fp)


print(f"len embeddings: {len(embeddings)}")

# Convert the labels to a Tensor
labels_section = torch.tensor(labels_section, dtype=torch.long)

# Now build and train the network
trainer = FlatClassifierModelTrainer(training_parameters, device)


######SUBMODEL2 - predicting full code now
print(f"embeddings type: {type(embeddings)}")
# Filter based on labels_section == 2
mask = torch.isin(labels_section, torch.tensor([2]))
embeddings1 = embeddings[mask]

print(f"labels_section: {type(labels_section)}")

labels1 = []
for label, section in zip(labels, labels_section):
    if section==2:
        labels1.append(label)


#texts1 = []
#for text, section in zip(texts, labels_section):
#    if section==2:
#        texts1.append(text) #Don't need this yet but will for multi_items tests later
        

print(f"labels1 values: {labels1[0:3]}")
print(f"embeddings1 type: {type(embeddings1)}")

##map subheadings for this subset only:
unique_labels1 = list(dict.fromkeys(labels1))
print(f"max of unique_labels: {max(unique_labels1)}, min: {min(unique_labels1)}")
matched_subheadings1 = [subheadings[label] for label in unique_labels1]
print(f"len unique_labels1 is {len(unique_labels1)}, len matched_subheadings1 is {len(matched_subheadings1)}, embeddings1: {len(embeddings1)}, labels1: {len(labels1)}")


##create dictionary from unique_labels1:
uniquelabels1dict={index: value for index, value in enumerate(unique_labels1)}

##Now use dict to map index in place of values in labels1:
labels1_encoded=[key for value in labels1 for key, dict_value in uniquelabels1dict.items() if dict_value == value]

##Need to turn into a tensor?
print(f"labels1_encoded type: {type(labels1_encoded)}")


matched_subheadings1_file=target_dir / "matched_subheadings1.pkl"

print("💾⇦ Saving matched_subheadings1")
with open(matched_subheadings1_file, "wb") as fp:
    pickle.dump(matched_subheadings1, fp)

        
# Convert the labels to a Tensor
labels1_encoded = torch.tensor(labels1_encoded, dtype=torch.long)

print(f"labels1_encoded: {type(labels1_encoded)}")
print(f"labels1_encoded values: {labels1_encoded[0:3]}")

print(embeddings1.size(0))
print(labels1_encoded.size())

###Don't think we need to add a vague terms category to each of them as that will be caught by the main Section model
labels1_file = data_dir / "labels1_encoded.pkl"
print("💾⇦ Saving labels0")
with open(labels1_file, "wb") as fp:
    pickle.dump(labels1_encoded, fp)


embeddings1_file = data_dir / "embeddings1.pkl"
print("💾⇦ Saving embeddings1")
with open(embeddings1_file, "wb") as fp:
    pickle.dump(embeddings1, fp)
    

print(f"learning rate is {args.learning_rate} and max_epochs is {args.max_epochs}")

model1 = trainer.run(embeddings1, labels1_encoded, len(matched_subheadings1)) 

print("💾⇦ Saving model1")

model_file1 = target_dir / "model2.pt"
torch.save(model1, model_file1)

print("✅ Training complete. Enjoy your model!")
