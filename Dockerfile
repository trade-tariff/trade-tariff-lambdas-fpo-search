FROM public.ecr.aws/lambda/python:3.11

COPY . ${LAMBDA_TASK_ROOT}

ENV SENTENCE_TRANSFORMERS_HOME=${LAMBDA_TASK_ROOT}/.sentence_transformer_cache/sentence_transformers/
ENV SENTENCE_TRANSFORMER_PRETRAINED_MODEL=all-mpnet-base-v2
ENV HF_HOME=${LAMBDA_TASK_ROOT}/.sentence_transformer_cache/transformers_cache/

RUN pip install --upgrade pip

RUN pip install accelerate --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements_lambda.txt

RUN python download_transformer.py

ENV OFFLINE=1

CMD ["handler.handle"]
