#!/usr/bin/env bash
# Build with Cloud Build and deploy to Cloud Run. The LLM key lives in Secret Manager, never in the image.
#   echo -n "$OPENAI_API_KEY" | gcloud secrets create docpilot-openai-key --data-file=-
set -euo pipefail
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"; REGION="${REGION:-asia-south1}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/docpilot/docpilot:$(git rev-parse --short HEAD)"
gcloud artifacts repositories create docpilot --repository-format=docker --location="$REGION" 2>/dev/null || true
gcloud builds submit --tag "$IMAGE" .
gcloud run deploy docpilot --image "$IMAGE" --region "$REGION" --port 8080 --cpu 2 --memory 2Gi \
  --min-instances 0 --max-instances 3 --concurrency 8 --allow-unauthenticated \
  --set-env-vars DOCPILOT_LLM_PROVIDER=openai,DOCPILOT_LLM_MODEL=gpt-4o-mini \
  --set-secrets OPENAI_API_KEY=docpilot-openai-key:latest
