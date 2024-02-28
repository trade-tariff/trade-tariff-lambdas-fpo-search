resource "aws_instance" "this" {
  ami           = data.aws_ami.this.id
  instance_type = "trn1.2xlarge"

  user_data                   = local.user_data
  user_data_replace_on_change = true

  associate_public_ip_address = true
  subnet_id                   = module.vpc.public_subnets[0]
  vpc_security_group_ids      = [aws_security_group.this.id]
  iam_instance_profile        = aws_iam_instance_profile.this.name

  key_name = data.aws_key_pair.this.key_name

  root_block_device {
    delete_on_termination = true
    volume_size           = 100
  }

  lifecycle {
    ignore_changes = [
      associate_public_ip_address # because stopping it dissociates the IP
    ]
  }
}

# we want the instance to be controlled by AWS CLI API calls, so we should keep
# it stopped in Terraform
resource "aws_ec2_instance_state" "this" {
  count       = var.stop_instance ? 1 : 0
  instance_id = aws_instance.this.id
  state       = "stopped"
  force       = false
}

module "bucket" {
  source = "github.com/terraform-aws-modules/terraform-aws-s3-bucket?ref=v4.1.0"

  bucket = "trade-tariff-fpo-model-data-${local.account_id}"
  acl    = "private"

  control_object_ownership = true
  object_ownership         = "ObjectWriter"

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_ssm_parameter" "github_token" {
  name  = "GITHUB_TOKEN"
  type  = "SecureString"
  value = var.github_token
}

variable "github_token" {
  description = "Value of GITHUB_TOKEN."
  type        = string
}

variable "stop_instance" {
  description = "Whether to stop the EC2 instance. Defaults to `false`."
  type        = bool
  default     = false
}
