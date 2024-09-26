##Note: submodel0 is simpler than the rest (as its labels already start from 0)

from fpo_args_parser import FPOArgsParser
import logging
from pathlib import Path
import pickle

import torch
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

    #print("💾⇨ Datafile pickle file found. Loading...")
    #with open(datafile_file, "rb") as fp:
    #    datafile = pickle.load(fp)

    print("💾⇨ Subheadings pickle file found. Loading...")
    with open(subheadings_file, "rb") as fp:
        subheadings = pickle.load(fp)



print(f"length of texts is {len(texts)}, length of labels is {len(labels)}, length of labels_section is {len(labels_section)}, max labels is {max(labels)}, max labels_section is {max(labels_section)}")

print(f"texts[0]: {texts[0]}")


print(f"labels type: {type(labels)}")
print(f"labels values: {labels[0:3]}")



# Next create the embeddings
print("Creating the embeddings")

embeddings_file = data_dir / "embeddings.pkl"

if not force and embeddings_file.exists():
    print("💾⇨ Embeddings pickle file found. Loading...")
    with open(embeddings_file, "rb") as fp:
        embeddings = pickle.load(fp)


print(f"len embeddings: {len(embeddings)}")

# Convert the labels to a Tensor
labels_section = torch.tensor(labels_section, dtype=torch.long) #needs to be a tensor in order to filter embeddings with it



# Now build and train the network
trainer = FlatClassifierModelTrainer(training_parameters, device)


######SUBMODEL0 - predicting full code now
print(f"embeddings type: {type(embeddings)}")
# Filter based on labels_section == 0
mask = torch.isin(labels_section, torch.tensor([0]))
embeddings0 = embeddings[mask]

print(f"labels_section: {type(labels_section)}")

labels0 = []
for label, section in zip(labels, labels_section):
    if section==0:
        labels0.append(label)

# Convert tensor to a list
print(f"labels0 values: {labels0[0:3]}")
print(f"embeddings0 type: {type(embeddings0)}")



##map subheadings for this subset only:
unique_labels0 = set(labels0)
matched_subheadings0 = [subheadings[label] for label in unique_labels0]
print(f"len unique_labels0 is {len(unique_labels0)}, len matched_subheadings0 is {len(matched_subheadings0)}, embeddings0: {len(embeddings0)}, labels0: {len(labels0)}")

matched_subheadings0_file=target_dir / "matched_subheadings0.pkl"

print("💾⇦ Saving matched_subheadings0")
with open(matched_subheadings0_file, "wb") as fp:
    pickle.dump(matched_subheadings0, fp)

        
# Convert the labels to a Tensor
labels0 = torch.tensor(labels0, dtype=torch.long)

print(embeddings0.size(0))
print(labels0.size())

###Don't think we need to add a vague terms category to each of them as that will be caught by the main Section model
labels0_file = data_dir / "labels0.pkl"
print("💾⇦ Saving labels0")
with open(labels0_file, "wb") as fp:
    pickle.dump(labels0, fp)


embeddings0_file = data_dir / "embeddings0.pkl"
print("💾⇦ Saving embeddings0")
with open(embeddings0_file, "wb") as fp:
    pickle.dump(embeddings0, fp)


print(f"learning rate is {args.learning_rate} and max_epochs is {args.max_epochs}")


model0 = trainer.run(embeddings0, labels0, len(matched_subheadings0)) #Ensure digits are set to 8 in fpo_args_parser

print("💾⇦ Saving model0")

model_file0 = target_dir / "model0.pt"
torch.save(model0, model_file0)

print("✅ Training complete. Enjoy your model!")
