#!/bin/bash

GIT_REPO="trade-tariff-lambdas-fpo-search"
S3_BUCKET_URI="s3://trade-tariff-fpo-model-data-382373577178"
DATE="$(date -u +%Y-%m-%d_%H%M%S)"
LOG_FILE="fpo-training-log-${DATE}.log"
RUNNING_IN_AWS=false
PYTHON_VERSION="3.11"
PROJECT_ROOT="/opt/fpo"

export PYENV_ROOT="${PROJECT_ROOT}/.pyenv"

mkdir -p "${PROJECT_ROOT}"
chmod 0755 "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}" || exit 1
find . -type f -name 'fpo-training-log-*' -delete
exec &> >(tee "${LOG_FILE}")

# test if we're on an EC2 instance
if curl -s -m 5 http://169.254.169.254/latest/dynamic/instance-identity/document | grep -q availabilityZone ; then
  RUNNING_IN_AWS=true
fi

if aws --version | grep "aws-cli/1"; then
  echo -e "\nUpdating AWS CLI.\n"
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  ./aws/install
fi

if ! which pyenv; then
  curl https://pyenv.run | bash
  command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
  yum remove openssl-devel -y
  yum install -y \
    gcc               \
    make              \
    patch             \
    zlib-devel        \
    bzip2             \
    bzip2-devel       \
    readline-devel    \
    sqlite            \
    sqlite-devel      \
    openssl11-devel   \
    tk-devel          \
    libffi-devel      \
    xz-devel
  pyenv install "${PYTHON_VERSION}"
  pyenv global "${PYTHON_VERSION}"
fi

if [ ! -d "${GIT_REPO}" ]; then
  GITHUB_TOKEN=$(aws ssm get-parameter \
  --region us-east-1                   \
  --name GITHUB_TOKEN                  \
  --with-decryption                    \
  --output text                        \
  --color off                          \
  --query Parameter.Value)

  echo -e "\nRepository not found!\n"
  git clone "https://trade-tariff-bot:${GITHUB_TOKEN}@github.com/trade-tariff/${GIT_REPO}.git"
fi

cd ${GIT_REPO} || exit 1

if [ ! -d "venv" ]; then
  echo -e "\nSetting up virtual python environment.\n"
  python -m venv venv
  make dev-env
fi

if [ ! -d "raw_source_data" ]; then
  echo -e "\nRetrieving raw source data.\n"
  mkdir raw_source_data
  aws s3 cp "${S3_BUCKET_URI}/raw_source_data/" raw_source_data/ --recursive
fi

echo -e "\nStarting model training.\n"

venv/bin/python${PYTHON_VERSION} train.py --force

if [ ! -d "benchmarking_data" ]; then
  echo -e "\nRetrieving benchmarking data.\n"
  mkdir benchmarking_data
  aws s3 cp "${S3_BUCKET_URI}/benchmarking_data/" benchmarking_data/ --recursive
fi

echo -e "\nStarting model benchmarking.\n"

venv/bin/python${PYTHON_VERSION} benchmark.py \
  --output json \
  --write-to-file

echo -e "\nUploading trained model to S3.\n"
aws s3 sync target/ "${S3_BUCKET_URI}/target/"

aws s3 cp \
  "benchmarking_data/results/benchmarking_results*.json" \
  "${S3_BUCKET_URI}benchmarking_data/results/" \
  --recursive

if $RUNNING_IN_AWS; then
  shutdown -h
else
  exit 0
fi
