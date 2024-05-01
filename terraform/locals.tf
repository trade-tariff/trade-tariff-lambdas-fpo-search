locals {
  region      = data.aws_region.current.name
  bucket_name = "trade-tariff-models-${data.aws_caller_identity.current.account_id}"
}
