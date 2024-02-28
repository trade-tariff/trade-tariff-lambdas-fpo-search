data "aws_iam_policy_document" "instance_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "fpo" {
  name               = "fpo-role"
  assume_role_policy = data.aws_iam_policy_document.instance_assume_role_policy.json
}

data "aws_iam_policy_document" "this" {
  statement {
    effect = "Allow"
    actions = [
      "s3:DeleteObject",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject",
    ]
    resources = [
      module.bucket.s3_bucket_arn,
      "${module.bucket.s3_bucket_arn}/*"
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "ssm:DescribeParameters",
      "ssm:GetParameter",
      "ssm:GetParameters"
    ]
    resources = [
      aws_ssm_parameter.github_token.arn
    ]
  }
}

resource "aws_iam_policy" "this" {
  name   = "fpo-policy"
  policy = data.aws_iam_policy_document.this.json
}

resource "aws_iam_role_policy_attachment" "this" {
  role       = aws_iam_role.fpo.name
  policy_arn = aws_iam_policy.this.arn
}

resource "aws_iam_instance_profile" "this" {
  name = "fpo_profile"
  role = aws_iam_role.fpo.name
}
