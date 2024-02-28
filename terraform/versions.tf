terraform {
  required_version = "~> 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5"
    }
  }

  backend "s3" {
    bucket = "terraform-state-production-382373577178"
    key    = "tariff-fpo.tfstate"
    region = "eu-west-2" # bucket is here
  }
}

provider "aws" {
  region = "us-east-1" # to access Trn1 instances
  default_tags {
    tags = {
      BillingCode = "HMR:WF"
    }
  }
}
