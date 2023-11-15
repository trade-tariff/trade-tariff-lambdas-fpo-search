from sentence_transformers import SentenceTransformer

# Just a simple script to download the Sentence Transformer model so that they can be baked into the docker build for quicker startup
SentenceTransformer("all-MiniLM-L6-v2")
