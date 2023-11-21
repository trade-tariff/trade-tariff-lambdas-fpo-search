from sentence_transformers import SentenceTransformer

import os

transformer = os.environ.get(
    "SENTENCE_TRANSFORMER_PRETRAINED_MODEL", "all-MiniLM-L6-v2"
)

# Just a simple script to download the Sentence Transformer model so that they can be baked into the docker build for quicker startup
SentenceTransformer(transformer)
