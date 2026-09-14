#!/usr/bin/env bash
set -Eeuo pipefail

required_variables=(
  AZURE_RESOURCE_GROUP
  AZURE_ML_WORKSPACE
  AZURE_ML_EXTENSION_VERSION
  DATA_ASSET
  MODEL_NAME
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "$variable_name is required" >&2
    exit 1
  fi
done

if [[ ! "$DATA_ASSET" =~ ^[A-Za-z0-9_.-]+:[1-9][0-9]*$ ]]; then
  echo "DATA_ASSET must use name:version format" >&2
  exit 1
fi
if [[ ! "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "MODEL_NAME contains unsupported characters" >&2
  exit 1
fi

data_name="${DATA_ASSET%:*}"
data_version="${DATA_ASSET##*:}"
common_args=(
  --resource-group "$AZURE_RESOURCE_GROUP"
  --workspace-name "$AZURE_ML_WORKSPACE"
)
train_job_file="${TRAIN_JOB_FILE:-azureml/jobs/train.yml}"
training_profile="${TRAINING_PROFILE:-production}"
base_model="${BASE_MODEL:-yolo11n.pt}"
train_compute_name="${TRAIN_COMPUTE_NAME:-}"
cpu_compute_name="${CPU_COMPUTE_NAME:-yolo-batch-cpu}"
if [[ ! "$training_profile" =~ ^(test|production)$ ]]; then
  echo "TRAINING_PROFILE must be test or production" >&2
  exit 1
fi
case "$train_job_file" in
  azureml/jobs/train.yml|azureml/jobs/continuous-train-test.yml) ;;
  *)
    echo "TRAIN_JOB_FILE is not an approved training definition" >&2
    exit 1
    ;;
esac
if [[ ! -f "$train_job_file" ]]; then
  echo "Training definition does not exist: $train_job_file" >&2
  exit 1
fi
if [[ -n "$train_compute_name" && ! "$train_compute_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "TRAIN_COMPUTE_NAME contains unsupported characters" >&2
  exit 1
fi
if [[ ! "$cpu_compute_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "CPU_COMPUTE_NAME contains unsupported characters" >&2
  exit 1
fi

if [[ "${SKIP_AZURE_ML_EXTENSION_INSTALL:-false}" != "true" ]]; then
  az extension add \
    --name ml \
    --version "$AZURE_ML_EXTENSION_VERSION" \
    --upgrade \
    --yes
fi

az ml data show \
  --name "$data_name" \
  --version "$data_version" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none

job_suffix="${JOB_NAME_SUFFIX:-manual}"
if [[ ! "$job_suffix" =~ ^[A-Za-z0-9-]+$ ]]; then
  echo "JOB_NAME_SUFFIX contains unsupported characters" >&2
  exit 1
fi

if [[ "${SKIP_DATA_VALIDATION:-false}" != "true" ]]; then
  validation_job="yolo-validate-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$job_suffix"
  az ml job create \
    --file azureml/jobs/validate-data.yml \
    --name "$validation_job" \
    --set inputs.training_data.path="azureml:$DATA_ASSET" \
      compute="azureml:$cpu_compute_name" \
      tags.training_data="$DATA_ASSET" \
      tags.trigger=direct-retraining \
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
fi

job_name="yolo-train-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$job_suffix"
job_overrides=(
  inputs.training_data.path="azureml:$DATA_ASSET"
  inputs.base_model="$base_model"
)
if [[ -n "$train_compute_name" ]]; then
  job_overrides+=(compute="azureml:$train_compute_name")
fi
az ml job create \
  --file "$train_job_file" \
  --name "$job_name" \
  --set "${job_overrides[@]}" \
  "${common_args[@]}" \
  --only-show-errors \
  --output none

az ml job stream --name "$job_name" "${common_args[@]}"

job_status=$(az ml job show \
  --name "$job_name" \
  "${common_args[@]}" \
  --query status \
  --output tsv)
if [[ "$job_status" != "Completed" ]]; then
  echo "Training job $job_name finished with status $job_status" >&2
  exit 1
fi

model_version=$(az ml model create \
  --name "$MODEL_NAME" \
  --type custom_model \
  --path "azureml://jobs/$job_name/outputs/model_output/paths/model.pt" \
  --tags source_job="$job_name" \
    training_data="$DATA_ASSET" \
    training_profile="$training_profile" \
  "${common_args[@]}" \
  --query version \
  --output tsv)

registered_lineage=$(az ml model show \
  --name "$MODEL_NAME" \
  --version "$model_version" \
  "${common_args[@]}" \
  --query "[tags.source_job, tags.training_data, tags.training_profile]" \
  --output tsv)
expected_lineage=$(printf '%s\t%s\t%s' "$job_name" "$DATA_ASSET" "$training_profile")
if [[ "$registered_lineage" != "$expected_lineage" ]]; then
  echo "Registered model lineage tags do not match the training run" >&2
  exit 1
fi

echo "Registered $MODEL_NAME:$model_version from $DATA_ASSET"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "job_name=$job_name" >> "$GITHUB_OUTPUT"
  echo "model_version=$model_version" >> "$GITHUB_OUTPUT"
fi
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "## Training result"
    echo
    echo "| Field | Value |"
    echo "| --- | --- |"
    echo "| Job | \`$job_name\` |"
    echo "| Data asset | \`$DATA_ASSET\` |"
    echo "| Registered model | \`$MODEL_NAME:$model_version\` |"
  } >> "$GITHUB_STEP_SUMMARY"
fi