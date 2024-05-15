from sentence_transformers import SentenceTransformer
from train_args import TrainScriptArgsParser

args = TrainScriptArgsParser()

# Just a simple script to download the Sentence Transformer model so that they can be baked into the docker build for quicker startup
SentenceTransformer(args.transformer())
