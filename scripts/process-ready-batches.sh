#!/usr/bin/env bash
set -Eeuo pipefail

required_variables=(
  AZURE_RESOURCE_GROUP
  AZURE_ML_WORKSPACE
  AZURE_ML_EXTENSION_VERSION
  STORAGE_ACCOUNT_NAME
  STORAGE_CONTAINER_NAME
  AML_DATASTORE_NAME
  DATA_ASSET_NAME
  MODEL_NAME
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "$variable_name is required" >&2
    exit 1
  fi
done

storage_prefix="${STORAGE_PREFIX:-continuous-training}"
max_batches="${MAX_BATCHES_PER_RUN:-1}"
training_profile="${TRAINING_PROFILE:-production}"
base_model_name="$MODEL_NAME"
test_model_name="${TEST_MODEL_NAME:-$base_model_name-test}"
base_model="${BASE_MODEL:-yolo11n.pt}"
cpu_compute_name="${CPU_COMPUTE_NAME:-yolo-batch-cpu}"
gpu_compute_name="${GPU_COMPUTE_NAME:-yolo-gpu-cluster}"
if [[ ! "$max_batches" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_BATCHES_PER_RUN must be a positive integer" >&2
  exit 1
fi
case "$training_profile" in
  test)
    train_job_file="azureml/jobs/continuous-train-test.yml"
    train_compute_name="$cpu_compute_name"
    ;;
  production)
    train_job_file="azureml/jobs/train.yml"
    train_compute_name="$gpu_compute_name"
    ;;
  *)
    echo "TRAINING_PROFILE must be test or production" >&2
    exit 1
    ;;
esac

common_args=(
  --resource-group "$AZURE_RESOURCE_GROUP"
  --workspace-name "$AZURE_ML_WORKSPACE"
)
az extension add \
  --name ml \
  --version "$AZURE_ML_EXTENSION_VERSION" \
  --upgrade \
  --yes
az ml datastore show \
  --name "$AML_DATASTORE_NAME" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none

mapfile -t markers < <(az storage blob list \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --container-name "$STORAGE_CONTAINER_NAME" \
  --prefix "$storage_prefix/" \
  --auth-mode login \
  --query "[?ends_with(name, '/_READY.json')].name" \
  --output tsv | sort)

if (( ${#markers[@]} == 0 )); then
  echo "No READY markers found under $storage_prefix/"
  exit 0
fi

processed_count=0
for marker_name in "${markers[@]}"; do
  marker_file=$(mktemp)
  trap 'rm -f "$marker_file"' EXIT
  az storage blob download \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --container-name "$STORAGE_CONTAINER_NAME" \
    --name "$marker_name" \
    --file "$marker_file" \
    --auth-mode login \
    --overwrite \
    --only-show-errors \
    --output none

    mapfile -t marker_values < <(
    python src/validate_marker.py "$marker_file" "$marker_name"
    )
  rm -f "$marker_file"
  trap - EXIT

  dataset_version="${marker_values[0]}"
  dataset_path="${marker_values[1]}"
  data_asset="$DATA_ASSET_NAME:$dataset_version"
  if [[ "$training_profile" == "test" ]]; then
    run_model_name="$test_model_name"
  else
    run_model_name="$base_model_name"
  fi

  existing_model=$(az ml model list \
    --name "$run_model_name" \
    "${common_args[@]}" \
    --query "[?tags.training_data=='$data_asset' && tags.training_profile=='$training_profile'].version | [0]" \
    --output tsv 2>/dev/null || true)
  if [[ -n "$existing_model" ]]; then
    echo "Skipping $data_asset; $run_model_name:$existing_model already uses it"
    continue
  fi

  if ! az ml data show \
    --name "$DATA_ASSET_NAME" \
    --version "$dataset_version" \
    "${common_args[@]}" \
    --only-show-errors \
    --output none 2>/dev/null; then
    az ml data create \
      --name "$DATA_ASSET_NAME" \
      --version "$dataset_version" \
      --type uri_folder \
      --path "azureml://datastores/$AML_DATASTORE_NAME/paths/$dataset_path" \
      --set tags.trigger=ready-marker tags.marker="$marker_name" \
      "${common_args[@]}" \
      --only-show-errors \
      --output none
  fi

  validation_job="yolo-validate-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-data-$dataset_version"
  az ml job create \
    --file azureml/jobs/validate-data.yml \
    --name "$validation_job" \
    --set inputs.training_data.path="azureml:$data_asset" \
      compute="azureml:$cpu_compute_name" \
      tags.training_data="$data_asset" \
      tags.trigger=ready-marker \
    "${common_args[@]}" \
    --only-show-errors \
    --output none
  az ml job stream --name "$validation_job" "${common_args[@]}"
  validation_status=$(az ml job show \
    --name "$validation_job" \
    "${common_args[@]}" \
    --query status \
    --output tsv)
  if [[ "$validation_status" != "Completed" ]]; then
    echo "Dataset validation failed with status $validation_status" >&2
    exit 1
  fi

  export DATA_ASSET="$data_asset"
  export MODEL_NAME="$run_model_name"
  export JOB_NAME_SUFFIX="$training_profile-data-$dataset_version"
  export SKIP_AZURE_ML_EXTENSION_INSTALL="true"
  export TRAIN_JOB_FILE="$train_job_file"
  export BASE_MODEL="$base_model"
  export TRAIN_COMPUTE_NAME="$train_compute_name"
  bash scripts/retrain.sh
  processed_count=$((processed_count + 1))
  if (( processed_count >= max_batches )); then
    break
  fi
done

echo "Processed $processed_count new dataset batch(es)"