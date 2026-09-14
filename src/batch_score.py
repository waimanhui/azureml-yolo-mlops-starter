import json
import os
from pathlib import Path

from ultralytics import YOLO


def find_model_path(model_dir: Path) -> Path:
    for filename in ("model.pt", "best.pt"):
        candidates = sorted(model_dir.rglob(filename))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"Expected one {filename} under {model_dir}, found {len(candidates)}"
            )
    raise FileNotFoundError(f"No model.pt or best.pt found under {model_dir}")


def init() -> None:
    global model

    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])
    model = YOLO(find_model_path(model_dir))


def run(mini_batch: list[str]) -> list[str]:
    output = []
    for image_path in mini_batch:
        try:
            result = model.predict(image_path, verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for coordinates, confidence, class_id in zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                ):
                    class_index = int(class_id)
                    detections.append(
                        {
                            "box_xyxy": coordinates,
                            "confidence": confidence,
                            "class_id": class_index,
                            "class_name": result.names[class_index],
                        }
                    )
            record = {
                "image": Path(image_path).name,
                "status": "ok",
                "detections": detections,
            }
        except Exception as error:
            record = {
                "image": Path(image_path).name,
                "status": "error",
                "error": str(error),
            }
        output.append(json.dumps(record))
    return output