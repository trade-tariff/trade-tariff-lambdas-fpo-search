#!/bin/bash

ssm_parameter="$1"

docker_tag=$(git rev-parse --short HEAD)
container="fpo-search:${docker_tag}"

function fetch_ecr_url {
  json=$(aws ssm get-parameter \
  --name "${ssm_parameter}"    \
  --with-decryption            \
  --output json                \
  --color off)

  output=$(jq -r .Parameter.Value <<< "${json}")

  if [ -n "${output}" ]; then
    echo "${output}"
  else
    exit 1
  fi
}

ecr_url=$(fetch_ecr_url)
account_id=$(aws sts get-caller-identity --output text --query 'Account')

aws s3 cp s3://trade-tariff-models-${account_id}/target.zip .
unzip target.zip
rm target.zip

git rev-parse --short HEAD > REVISION

aws s3 cp s3://trade-tariff-models-${account_id}/spelling-data.zip .
unzip spelling-data.zip
rm spelling-data.zip

docker build -t "$container" .
docker tag "${container}" "${ecr_url}:${docker_tag}"

aws ecr get-login-password | docker login --username AWS --password-stdin "${ecr_url}"

docker push "${ecr_url}:${docker_tag}"
