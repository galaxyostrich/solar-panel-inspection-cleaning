from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def add_local_import_paths() -> None:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    for path in (script_dir, cwd):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_default_model() -> Path:
    candidates = [
        Path.cwd() / "best.pt",
        Path(__file__).resolve().parent / "best.pt",
        Path(__file__).resolve().parent.parent / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / "best.pt"


def load_yolo_model(model_path: Path):
    from ultralytics import YOLO

    return YOLO(str(model_path))


def imwrite_unicode(path: Path, image: Any) -> bool:
    import cv2
    import numpy as np

    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def write_yolo_labels(result: Any, label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if result.boxes is None or len(result.boxes) == 0:
        label_path.write_text("", encoding="utf-8")
        return

    xywhn = result.boxes.xywhn.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    lines = [
        f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
        for (cx, cy, width, height), class_id in zip(xywhn, classes)
    ]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def predict_and_write_outputs(
    model: Any,
    image_path: Path,
    label_path: Path,
    overlay_path: Path,
    conf: float,
    iou: float,
) -> None:
    result = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)[0]
    write_yolo_labels(result, label_path)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(overlay_path, result.plot()):
        raise RuntimeError(f"Failed to write overlay image: {overlay_path}")


def choose_images(input_dir: Path, left_image: Path | None, right_image: Path | None) -> list[Path]:
    if left_image is not None and right_image is not None:
        return [left_image, right_image]

    images = list_images(input_dir)
    if len(images) < 2:
        raise ValueError(
            f"Expected at least 2 images in {input_dir}, found {len(images)}. "
            "Use --left-image and --right-image to specify them explicitly."
        )
    return images


def run_pair(
    input_dir: Path,
    model_path: Path,
    left_image: Path | None = None,
    right_image: Path | None = None,
    out_dir: Path | None = None,
    conf: float = 0.1,
    iou: float = 0.5,
    out_width: int = 1600,
    out_height: int = 900,
) -> list[tuple[Path, Path]]:
    add_local_import_paths()
    import fix_2_preprocess

    input_dir = input_dir.resolve()
    out_dir = (out_dir or (input_dir / "outputs")).resolve()
    processed_dir = out_dir / "processed"
    meta_dir = out_dir / "processed_meta"
    label_dir = out_dir / "labels"
    overlay_dir = out_dir / "overlays"
    processed_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [path.resolve() for path in choose_images(input_dir, left_image, right_image)]

    model = load_yolo_model(model_path.resolve())
    if len(image_paths) > len(fix_2_preprocess.POINTS_BY_POSITION):
        raise ValueError(
            f"Found {len(image_paths)} images but only "
            f"{len(fix_2_preprocess.POINTS_BY_POSITION)} fixed point sets"
        )

    jobs = []
    for idx, src_path in enumerate(image_paths):
        side = f"tile{idx + 1}"
        position = idx + 1
        jobs.append((side, src_path, position))

    output_paths: list[tuple[Path, Path]] = []
    for side, src_path, position in jobs:
        result = fix_2_preprocess.process_one_image(
            src_path,
            fix_2_preprocess.POINTS_BY_POSITION[position],
            out_w=out_width,
            out_h=out_height,
            out_dir=processed_dir,
            meta_dir=meta_dir,
            position=position,
        )
        if result is None:
            raise RuntimeError(f"Failed to preprocess {side} image: {src_path}")

        processed_path, _meta_path = result
        label_path = label_dir / f"{src_path.stem}_{side}.txt"
        overlay_path = overlay_dir / f"{src_path.stem}_{side}_pred.png"
        predict_and_write_outputs(model, processed_path, label_path, overlay_path, conf=conf, iou=iou)
        output_paths.append((label_path, overlay_path))
        print(f"{side}: {src_path.name} -> {processed_path.name} -> {label_path.name}, {overlay_path.name}")

    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess up to four images with fix_2 fixed perspective points, then run best.pt detection."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing the source images.")
    parser.add_argument("--model", type=Path, default=None, help="Path to best.pt. Defaults to ./best.pt.")
    parser.add_argument("--left-image", type=Path, default=None, help="Explicit left image path.")
    parser.add_argument("--right-image", type=Path, default=None, help="Explicit right image path.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder. Defaults to input_dir/outputs.")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold.")
    parser.add_argument("--out-width", type=int, default=1600, help="Preprocessed image width.")
    parser.add_argument("--out-height", type=int, default=900, help="Preprocessed image height.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = args.model or resolve_default_model()
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1
    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}")
        return 1

    try:
        output_paths = run_pair(
            input_dir=args.input_dir,
            model_path=model_path,
            left_image=args.left_image,
            right_image=args.right_image,
            out_dir=args.out_dir,
            conf=args.conf,
            iou=args.iou,
            out_width=args.out_width,
            out_height=args.out_height,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Label txt files:")
    for label_path, _overlay_path in output_paths:
        print(label_path)
    print("Overlay image files:")
    for _label_path, overlay_path in output_paths:
        print(overlay_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
