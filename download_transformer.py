from sentence_transformers import SentenceTransformer
from inference.infer import transformer


# Just a simple script to download the Sentence Transformer model so that they can be baked into the docker build for quicker startup
SentenceTransformer(transformer)
