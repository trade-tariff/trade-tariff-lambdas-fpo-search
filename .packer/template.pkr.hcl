packer {
  required_plugins {
    amazon = {
      version = ">= 0.0.2"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

source "amazon-ebs" "source" {
  ami_name      = "Deep Learning AMI Neuron PyTorch - {{timestamp}}"
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
    script = ".packer/provision"
  }
}
