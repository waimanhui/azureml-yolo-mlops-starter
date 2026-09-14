#!/usr/bin/env bash
set -Eeuo pipefail

required_variables=(STORAGE_ACCOUNT_NAME STORAGE_CONTAINER_NAME)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "$variable_name is required" >&2
    exit 1
  fi
done

snapshot_root="${SNAPSHOT_ROOT:-data/continuous-simulation/snapshots}"
arrival_delay_seconds="${ARRIVAL_DELAY_SECONDS:-0}"
if [[ ! "$arrival_delay_seconds" =~ ^[0-9]+$ ]]; then
  echo "ARRIVAL_DELAY_SECONDS must be a non-negative integer" >&2
  exit 1
fi
if [[ ! -d "$snapshot_root" ]]; then
  echo "Snapshot directory does not exist: $snapshot_root" >&2
  exit 1
fi

mapfile -t snapshot_dirs < <(find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d | sort)
if (( ${#snapshot_dirs[@]} == 0 )); then
  echo "No snapshots found under $snapshot_root" >&2
  exit 1
fi

for snapshot_dir in "${snapshot_dirs[@]}"; do
  marker_path="$snapshot_dir/_READY.json"
  if [[ ! -f "$marker_path" ]]; then
    echo "Missing marker: $marker_path" >&2
    exit 1
  fi
  dataset_path=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["dataset_path"])' \
    "$marker_path")
  if [[ ! "$dataset_path" =~ ^[A-Za-z0-9._/-]+$ ]] || [[ "$dataset_path" == *".."* ]]; then
    echo "Unsafe dataset_path in $marker_path" >&2
    exit 1
  fi

  payload_dir=$(mktemp -d)
  trap 'rm -rf "$payload_dir"' EXIT
  cp -R "$snapshot_dir/." "$payload_dir/"
  rm "$payload_dir/_READY.json"

  echo "Uploading dataset payload to $dataset_path"
  az storage blob upload-batch \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --destination "$STORAGE_CONTAINER_NAME" \
    --destination-path "$dataset_path" \
    --source "$payload_dir" \
    --auth-mode login \
    --overwrite false \
    --only-show-errors \
    --output none

  echo "Publishing completion marker last: $dataset_path/_READY.json"
  az storage blob upload \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --container-name "$STORAGE_CONTAINER_NAME" \
    --name "$dataset_path/_READY.json" \
    --file "$marker_path" \
    --auth-mode login \
    --overwrite false \
    --only-show-errors \
    --output none

  rm -rf "$payload_dir"
  trap - EXIT
  if (( arrival_delay_seconds > 0 )); then
    sleep "$arrival_delay_seconds"
  fi
done