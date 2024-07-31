FROM public.ecr.aws/lambda/python:3.11

COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements_lambda.txt

# Download the transformer model and then delete the hf cache version
RUN <<EOF
    python download_transformer.py
    rm -rf ~/.cache/torch/sentence_transformers
EOF

CMD ["handler.handle"]
