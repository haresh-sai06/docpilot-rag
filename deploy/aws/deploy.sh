#!/usr/bin/env bash
# Push to ECR and register an ECS Fargate task definition (put it behind an ALB / ECS service).
set -euo pipefail
REGION="${AWS_REGION:-ap-south-1}"; ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/docpilot"
aws ecr describe-repositories --repository-names docpilot --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name docpilot --region "$REGION"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%/*}"
docker build -t "$REPO:latest" . && docker push "$REPO:latest"
sed "s#ACCOUNT_ID#$ACCOUNT#g" deploy/aws/task-definition.json > /tmp/docpilot-task.json
aws ecs register-task-definition --cli-input-json file:///tmp/docpilot-task.json --region "$REGION"
echo "Now: aws ecs create-service --cluster <cluster> --service-name docpilot --task-definition docpilot --launch-type FARGATE ..."
