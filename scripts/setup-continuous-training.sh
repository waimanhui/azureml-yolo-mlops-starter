#!/usr/bin/env bash
set -Eeuo pipefail

required_variables=(
  AZURE_RESOURCE_GROUP
  AZURE_ML_WORKSPACE
  AZURE_ML_EXTENSION_VERSION
  AZURE_CLIENT_OBJECT_ID
  STORAGE_ACCOUNT_NAME
  STORAGE_CONTAINER_NAME
  AML_DATASTORE_NAME
  CPU_COMPUTE_NAME
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "$variable_name is required" >&2
    exit 1
  fi
done

az extension add \
  --name ml \
  --version "$AZURE_ML_EXTENSION_VERSION" \
  --upgrade \
  --yes

az deployment group create \
  --name "yolo-continuous-training-${GITHUB_RUN_ID:-local}" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --template-file infra/continuous-training.bicep \
  --parameters \
    workspaceName="$AZURE_ML_WORKSPACE" \
    storageAccountName="$STORAGE_ACCOUNT_NAME" \
    containerName="$STORAGE_CONTAINER_NAME" \
    githubPrincipalId="$AZURE_CLIENT_OBJECT_ID" \
    cpuComputeName="$CPU_COMPUTE_NAME" \
  --only-show-errors \
  --output none

common_args=(
  --resource-group "$AZURE_RESOURCE_GROUP"
  --workspace-name "$AZURE_ML_WORKSPACE"
)
if az ml datastore show \
  --name "$AML_DATASTORE_NAME" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none 2>/dev/null; then
  datastore_command=update
else
  datastore_command=create
fi

az ml datastore "$datastore_command" \
  --file azureml/datastores/continuous-training.yml \
  --set \
    name="$AML_DATASTORE_NAME" \
    account_name="$STORAGE_ACCOUNT_NAME" \
    container_name="$STORAGE_CONTAINER_NAME" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none

echo "Continuous-training storage and Azure ML datastore are ready"