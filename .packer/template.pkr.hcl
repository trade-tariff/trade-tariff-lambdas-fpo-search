packer {
  required_plugins {
    amazon = {
      version = ">= 0.0.2"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "ami_version" {
  type        = string
  description = "Our internally managed version for our own custom AMI. We bump this when we make changes to the AMI and use it in the name of the AMI."
}

variable "git_branch" {
  type        = string
  description = "The git branch to use to build dependencies from"
  default     = "main"
}

source "amazon-ebs" "source" {
  ami_name = "FPO Training ${var.ami_version}"
  ami_users = [
    "844815912454",
    "382373577178",
    "451934005581"
  ]

  instance_type = "p2.xlarge"
  ssh_username  = "ec2-user"
  region        = "us-east-1"

  source_ami = "ami-0b0307e6a5c41a255"
}

build {
  sources = ["source.amazon-ebs.source"]

  provisioner "shell" {
    script           = "provision"
    environment_vars = ["GIT_BRANCH=${var.git_branch}"]
  }
}
