FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    SENTENCE_TRANSFORMERS_HOME=/opt/app/.sentence_transformer_cache/sentence_transformers/ \
    SENTENCE_TRANSFORMER_PRETRAINED_MODEL=all-mpnet-base-v2 \
    HF_HOME=/opt/app/.sentence_transformer_cache/transformers_cache/ \
    OFFLINE=1

# Install Python dependencies
COPY requirements_lambda.txt /tmp/
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements_lambda.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /opt/app
WORKDIR /opt/app
RUN python download_transformer.py

ENTRYPOINT ["python3.11", "-m", "awslambdaric"]
CMD ["handler.handle"]
