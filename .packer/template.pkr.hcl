packer {
  required_plugins {
    amazon = {
      version = ">= 0.0.2"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "ami_version" {
  type = string
  description = "Our internally managed version for our own custom AMI. We bump this when we make changes to the AMI and use it in the name of the AMI."
}

source "amazon-ebs" "source" {
  ami_name      = "Deep Learning AMI Neuron PyTorch - ${var.ami_version}"
  ami_users     = [
    "844815912454",
    "382373577178",
    "451934005581"
  ]
  instance_type = "trn1.2xlarge"
  ssh_username = "ec2-user"
  region        = "us-east-1"
  source_ami_filter {
    filters = {
      name   = "Deep Learning AMI Neuron PyTorch 1.13*" # Pin to a specific version for now
      root-device-type = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["898082745236"]
  }
}

build {
  sources = ["source.amazon-ebs.source"]

  provisioner "shell" {
    script = "provision"
  }
}