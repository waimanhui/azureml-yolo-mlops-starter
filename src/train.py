import argparse
import shutil
from pathlib import Path


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


def validate_device(device: str) -> None:
    if device.strip().lower() == "cpu":
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested but CUDA is unavailable "
            f"(PyTorch {torch.__version__}, runtime {torch.version.cuda}, "
            f"visible devices {torch.cuda.device_count()}). Check that the "
            "container CUDA version supports the compute GPU and host driver."
        )


def main() -> None:
    args = parse_args()
    validate_device(args.device)

    from ultralytics import YOLO

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