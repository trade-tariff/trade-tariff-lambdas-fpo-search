FROM public.ecr.aws/lambda/python:3.11

COPY . ${LAMBDA_TASK_ROOT}

ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers/
ENV SENTENCE_TRANSFORMER_PRETRAINED_MODEL=all-mpnet-base-v2
ENV HF_HOME=/tmp/transformers_cache/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements_lambda.txt

RUN python download_transformer.py && \
  rm -f "$SENTENCE_TRANSFORMERS_HOME/sentence-transformers_${SENTENCE_TRANSFORMER_PRETRAINED_MODEL}/model.safetensors" # We use torch weight files

CMD ["handler.handle"]
