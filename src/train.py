import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.model_output)
    run_dir = output_dir / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        project=str(run_dir),
        name="run",
        exist_ok=True,
        plots=False,
    )

    best_model = run_dir / "run" / "weights" / "best.pt"
    if not best_model.exists():
        raise FileNotFoundError(f"Training did not produce {best_model}")
    shutil.copy2(best_model, output_dir / "model.pt")


if __name__ == "__main__":
    main()