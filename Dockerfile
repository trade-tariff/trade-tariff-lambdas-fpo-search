FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY requirements_lambda.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements_lambda.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir awslambdaric==3.1.1 && \
    pip cache purge && \
    rm -rf /root/.cache/pip

COPY . .

RUN python quantize_model.py

FROM python:3.12-slim AS production

WORKDIR /opt/app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/app .

ENV SENTENCE_TRANSFORMERS_HOME=/opt/app/.sentence_transformer_cache/sentence_transformers/ \
    SENTENCE_TRANSFORMER_PRETRAINED_MODEL=all-mpnet-base-v2 \
    HF_HOME=/opt/app/.sentence_transformer_cache/transformers_cache/ \
    OFFLINE=1

RUN python download_transformer.py && \
    rm -rf /root/.cache /opt/app/.sentence_transformer_cache/transformers_cache

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v1.27/aws-lambda-rie /usr/bin/aws-lambda-rie
RUN chmod 700 /usr/bin/aws-lambda-rie

ENTRYPOINT ["/opt/app/bin/entry"]
CMD ["handler.handle"]
