from pathlib import Path
from sentence_transformers import SentenceTransformer
from train_args import TrainScriptArgsParser

args = TrainScriptArgsParser()

dir = Path(args.transformer_cache_directory())
dir.mkdir(parents=True, exist_ok=True)

# Just a simple script to download the Sentence Transformer model so that they can be baked into the docker build for quicker startup
transformer_model = SentenceTransformer(args.transformer(), cache_folder=args.transformer_cache_directory())
