from data_sources.vague_terms import VagueTermsCSVDataSource
from train_args import TrainScriptArgsParser
import logging
from pathlib import Path

import pickle

import torch
from data_sources.search_references import SearchReferencesDataSource
from data_sources.data_source import DataSource
from data_sources.basic_csv import BasicCSVDataSource
from training.create_embeddings import EmbeddingsProcessor
from training.prepare_data import TrainingDataLoader
from training.train_model import (
    FlatClassifierModelTrainer,
    FlatClassifierModelTrainerParameters,
)

args = TrainScriptArgsParser()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

training_parameters = FlatClassifierModelTrainerParameters(
    args.learning_rate(), args.max_epochs()
)

print(f"⚙️  Using device {args.device()}")

cwd = Path(__file__).resolve().parent

target_dir = args.target_dir()
target_dir.mkdir(parents=True, exist_ok=True)

data_dir = args.data_dir()
data_dir.mkdir(parents=True, exist_ok=True)

# First load in the training data
print("💾⇨ Loading training data")

subheadings_file = target_dir / "subheadings.pkl"

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
        authoritative=True,
        creates_codes=True,
    )
)

data_sources += [
    BasicCSVDataSource(filename, encoding="latin_1")
    for filename in Path(args.tradesets_data_dir()).glob("*.csv")
]

training_data_loader = TrainingDataLoader()

(text_values, subheadings, texts, labels) = training_data_loader.fetch_data(
    data_sources, 8
)
print(f"Found {len(text_values)} unique descriptions")

print("💾⇦ Saving subheadings")
with open(subheadings_file, "wb") as fp:
    pickle.dump(subheadings, fp)

# Impose the limit if required - this will limit the number of unique descriptions
if args.limit() is not None:
    text_values = text_values[:args.limit()]

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
    cache_path=args.cache_dir(),
    torch_device=args.torch_device(),
    batch_size=args.embedding_batch_size(),
    cache_checkpoint=args.embedding_cache_checkpoint(),
)

unique_embeddings = embeddings_processor.create_embeddings(text_values)

# Now build and train the network
trainer = FlatClassifierModelTrainer(
    training_parameters, device=args.torch_device(), batch_size=args.model_batch_size()
)

# Convert the labels to a Tensor
labels = torch.tensor(labels, dtype=torch.long)

embeddings = torch.stack([unique_embeddings[idx] for idx in texts])

model = trainer.run(embeddings, labels, len(subheadings))

print("💾⇦ Saving model")

model_file = target_dir / "model.pt"
torch.save(model, model_file)

print("✅ Training complete. Enjoy your model!")
