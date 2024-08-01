FROM public.ecr.aws/lambda/python:3.12

COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements_lambda.txt

# Download the transformer model and then delete the hf cache version
RUN python download_transformer.py && rm -rf ~/.cache/torch/sentence_transformers

# Run an inference which should create most of the pyc files
RUN python infer.py 'plastic toothbrush'

CMD ["handler.handle"]
