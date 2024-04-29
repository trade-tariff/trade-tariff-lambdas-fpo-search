data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

data "aws_key_pair" "this" {
  key_name           = "fpo-training-instance"
  include_public_key = true
}

data "aws_s3_bucket" "this" {
  bucket = local.bucket_name
}
