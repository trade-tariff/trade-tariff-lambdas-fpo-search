resource "aws_instance" "this" {
  ami           = "ami-0e4a0478fee8d8aae"
  instance_type = "trn1.2xlarge"

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
  instance_id = aws_instance.this.id
  state       = "stopped"
  force       = false
}
