from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]


def read_script(name: str) -> str:
    return (REPOSITORY_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_retraining_validates_data_and_persists_model_lineage() -> None:
    script = read_script("retrain.sh")

    assert "azureml/jobs/validate-data.yml" in script
    assert "--tags source_job=" in script
    assert "Registered model lineage tags do not match" in script


def test_continuous_training_fails_closed_and_verifies_data_path() -> None:
    script = read_script("process-ready-batches.sh")

    assert "Unable to check whether $data_asset was already trained" in script
    assert "expected_data_suffix" in script
    assert "Production processing supports one batch" in script
    assert 'inputs.expected_train_images="$expected_train_images"' in script
    assert 'inputs.expected_val_images="$expected_val_images"' in script
    assert 'export SKIP_DATA_VALIDATION="true"' in script


def test_deployment_requires_production_lineage() -> None:
    script = read_script("deploy-batch.sh")

    assert "tags.training_profile" in script
    assert '-z "$source_job"' in script
    assert '-z "$training_data"' in script
    assert "missing verified production lineage tags" in script