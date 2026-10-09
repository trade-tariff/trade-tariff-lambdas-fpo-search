# Build the Lambda Runtime Interface Emulator from source. The upstream
# release binaries (v1.37 and earlier) use a Go toolchain older than 1.26.9,
# which has HIGH CVEs in net/http and crypto/tls (CVE-2026-78667,
# CVE-2026-97031). Move back to the release binary when upstream ships one
# that uses Go 1.26.9 or later.
FROM golang:1.26.9-trixie AS rie-builder

ARG RIE_VERSION=v1.37

WORKDIR /src

RUN git clone --depth 1 --branch ${RIE_VERSION} \
    https://github.com/aws/aws-lambda-runtime-interface-emulator.git . && \
    CGO_ENABLED=0 go build -buildvcs=false -ldflags "-s -w" \
    -o /usr/local/bin/aws-lambda-rie ./cmd/aws-lambda-rie

FROM python:3.14-slim AS builder

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY requirements.txt .
COPY requirements_lambda.txt .

# TODO: Revisit awslambdaric pin after upstream issue is fixed:
# https://github.com/aws/aws-lambda-nodejs-runtime-interface-client/issues/170
RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements_lambda.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir awslambdaric==3.1.1 && \
    pip cache purge && \
    rm -rf /root/.cache/pip

COPY . .

RUN python quantize_model.py

FROM python:3.14-slim AS production

RUN apt-get update && apt-get upgrade -y libssl3t64 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/app .

# pip vendors a CycloneDX SBOM naming the setuptools/msgpack versions its
# vendored code was sourced from; Trivy reads it as installed packages and
# flags their CVEs even though the vulnerable modules aren't present.
RUN rm -f /usr/local/lib/python3.14/site-packages/pip/_vendor/bom.cdx.json

ENV SENTENCE_TRANSFORMERS_HOME=/opt/app/.sentence_transformer_cache/sentence_transformers/ \
    SENTENCE_TRANSFORMER_PRETRAINED_MODEL=all-mpnet-base-v2 \
    HF_HOME=/opt/app/.sentence_transformer_cache/transformers_cache/ \
    OFFLINE=1

RUN python download_transformer.py && \
    rm -rf /root/.cache /opt/app/.sentence_transformer_cache/transformers_cache

COPY --from=rie-builder /usr/local/bin/aws-lambda-rie /usr/bin/aws-lambda-rie
RUN chmod 700 /usr/bin/aws-lambda-rie

ENTRYPOINT ["/opt/app/bin/entry"]
CMD ["handler.handle"]
