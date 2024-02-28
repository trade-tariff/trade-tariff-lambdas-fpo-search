#!/bin/bash

INSTANCE_ID=$(aws ec2 describe-instances          \
  --region us-east-1                              \
  --query Reservations[0].Instances[0].InstanceId \
  --output text
)

aws ec2 start-instances       \
  --instance-ids $INSTANCE_ID \
  --region us-east-1
