from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from train_args import TrainScriptArgsParser

args = TrainScriptArgsParser()

# Just a simple script to download the Sentence Transformer model so that they can be baked into the docker build for quicker startup
transformer_model = SentenceTransformer(args.transformer(), cache_folder=None)

dir = Path(args.transformer_cache_directory())
dir.mkdir(parents=True, exist_ok=True)

# Save as a .pt file as it loads an order of magnitude quicker than using the SentenceTransformer constructor
torch.save(
    transformer_model,
    Path(args.transformer_cache_directory()) / f"{args.transformer()}_transformer_model.pt",
)
