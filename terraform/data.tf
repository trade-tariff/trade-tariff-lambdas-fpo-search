data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# Deep Learning Base Neuron AMI (Amazon Linux 2)
data "aws_ami" "this" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI Neuron PyTorch*"]
  }

  filter {
    name   = "name"
    values = ["*(Amazon Linux 2)*"]
  }

  filter {
    name   = "description"
    values = ["*Trn1*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_key_pair" "this" {
  key_name           = "fpo-training-instance"
  include_public_key = true
}

locals {
  region     = data.aws_region.current.name
  account_id = data.aws_caller_identity.current.account_id
  user_data  = templatefile("${path.module}/user_data.tftpl", { script = file("${path.module}/script.sh") })
}
