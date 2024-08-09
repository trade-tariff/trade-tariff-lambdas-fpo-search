FROM public.ecr.aws/lambda/python:3.12

ENV SENTENCE_TRANSFORMERS_HOME=${LAMBDA_TASK_ROOT}/.sentence_transformer_cache/transformers
ENV HF_HOME=${LAMBDA_TASK_ROOT}/.sentence_transformer_cache/huggingface

COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements_lambda.txt

# Run an inference which should create most of the pyc files and cache the sentence transformer
RUN python infer.py 'plastic toothbrush'

CMD ["handler.handle"]
