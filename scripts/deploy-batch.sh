#!/usr/bin/env bash
set -Eeuo pipefail

required_variables=(
  AZURE_RESOURCE_GROUP
  AZURE_ML_WORKSPACE
  AZURE_ML_EXTENSION_VERSION
  MODEL_NAME
  MODEL_VERSION
  BATCH_ENDPOINT_NAME
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "$variable_name is required" >&2
    exit 1
  fi
done

if [[ ! "$MODEL_VERSION" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_VERSION must be a positive integer" >&2
  exit 1
fi
for resource_name in "$MODEL_NAME" "$BATCH_ENDPOINT_NAME"; do
  if [[ ! "$resource_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Azure ML resource name contains unsupported characters" >&2
    exit 1
  fi
done

common_args=(
  --resource-group "$AZURE_RESOURCE_GROUP"
  --workspace-name "$AZURE_ML_WORKSPACE"
)
deployment_name="model-v$MODEL_VERSION"
batch_compute_name="${BATCH_COMPUTE_NAME:-yolo-batch-cpu}"
if [[ ! "$batch_compute_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "BATCH_COMPUTE_NAME contains unsupported characters" >&2
  exit 1
fi

az extension add \
  --name ml \
  --version "$AZURE_ML_EXTENSION_VERSION" \
  --upgrade \
  --yes

model_lineage=$(az ml model show \
  --name "$MODEL_NAME" \
  --version "$MODEL_VERSION" \
  "${common_args[@]}" \
  --query "[tags.source_job, tags.training_data, tags.training_profile]" \
  --output tsv)
IFS=$'\t' read -r source_job training_data training_profile <<< "$model_lineage"
if [[ -z "$source_job" || -z "$training_data" || "$training_profile" != "production" ]]; then
  echo "$MODEL_NAME:$MODEL_VERSION is missing verified production lineage tags" >&2
  exit 1
fi

if ! az ml batch-endpoint show \
  --name "$BATCH_ENDPOINT_NAME" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none 2>/dev/null; then
  az ml batch-endpoint create \
    --file azureml/endpoints/batch-endpoint.yml \
    --set name="$BATCH_ENDPOINT_NAME" \
    "${common_args[@]}" \
    --only-show-errors \
    --output none
fi

if az ml batch-deployment show \
  --name "$deployment_name" \
  --endpoint-name "$BATCH_ENDPOINT_NAME" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none 2>/dev/null; then
  deployment_command=update
else
  deployment_command=create
fi

az ml batch-deployment "$deployment_command" \
  --file azureml/endpoints/batch-deployment.yml \
  --set name="$deployment_name" \
    endpoint_name="$BATCH_ENDPOINT_NAME" \
    model="azureml:$MODEL_NAME:$MODEL_VERSION" \
    compute="azureml:$batch_compute_name" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none

provisioning_state=$(az ml batch-deployment show \
  --name "$deployment_name" \
  --endpoint-name "$BATCH_ENDPOINT_NAME" \
  "${common_args[@]}" \
  --query provisioning_state \
  --output tsv)

if [[ "$provisioning_state" != "Succeeded" ]]; then
  echo "Deployment $deployment_name has state $provisioning_state" >&2
  exit 1
fi

az ml batch-endpoint update \
  --name "$BATCH_ENDPOINT_NAME" \
  --set defaults.deployment_name="$deployment_name" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none

default_deployment=$(az ml batch-endpoint show \
  --name "$BATCH_ENDPOINT_NAME" \
  "${common_args[@]}" \
  --query defaults.deployment_name \
  --output tsv)
if [[ "$default_deployment" != "$deployment_name" ]]; then
  echo "Endpoint default is $default_deployment, expected $deployment_name" >&2
  exit 1
fi

echo "Deployed $MODEL_NAME:$MODEL_VERSION as $BATCH_ENDPOINT_NAME/$deployment_name"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "## Production deployment"
    echo
    echo "| Field | Value |"
    echo "| --- | --- |"
    echo "| Endpoint | \`$BATCH_ENDPOINT_NAME\` |"
    echo "| Deployment | \`$deployment_name\` |"
    echo "| Model | \`$MODEL_NAME:$MODEL_VERSION\` |"
    echo "| Provisioning state | \`$provisioning_state\` |"
  } >> "$GITHUB_STEP_SUMMARY"
fi