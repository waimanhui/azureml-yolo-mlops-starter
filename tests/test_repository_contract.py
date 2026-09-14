import ast
import json
import re
from pathlib import Path

import nbformat
import yaml

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_yaml_files_parse() -> None:
    yaml_files = sorted(REPOSITORY_ROOT.rglob("*.yml"))

    assert yaml_files
    for yaml_file in yaml_files:
        yaml.safe_load(yaml_file.read_text(encoding="utf-8"))


def test_datastore_name_uses_supported_characters() -> None:
    datastore_path = REPOSITORY_ROOT / "azureml/datastores/continuous-training.yml"
    datastore = yaml.safe_load(datastore_path.read_text(encoding="utf-8"))

    assert re.fullmatch(r"[A-Za-z0-9_]+", datastore["name"])


def test_notebook_structure_and_python_syntax() -> None:
    notebook_path = REPOSITORY_ROOT / "yolo-training-test.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))

    assert notebook["cells"]
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        assert cell.get("id") or cell["metadata"].get("id")
        language = cell["metadata"].get("language")
        if language is None:
            language = "python" if cell["cell_type"] == "code" else "markdown"
        assert language in {"markdown", "python"}
        assert isinstance(cell["source"], list)
        if cell["cell_type"] != "code":
            continue

        source_lines = [
            f"{' ' * (len(line) - len(line.lstrip()))}pass"
            if line.lstrip().startswith("%")
            else line
            for line in cell["source"]
        ]
        ast.parse("\n".join(source_lines), filename=f"notebook cell {cell_number}")
