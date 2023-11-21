FROM public.ecr.aws/lambda/python:3.11

COPY . ${LAMBDA_TASK_ROOT}

ENV SENTENCE_TRANSFORMERS_HOME=${LAMBDA_TASK_ROOT}/tmp/sentence_transformers/
ENV SENTENCE_TRANSFORMER_PRETRAINED_MODEL=all-MiniLM-L6-v2
ENV TRANSFORMERS_CACHE=${LAMBDA_TASK_ROOT}/tmp/transformers_cache/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements_lambda.txt

RUN python download_transformer.py

CMD ["handler.handle"]
