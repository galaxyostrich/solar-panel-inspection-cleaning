from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from left_fixpreprocess import (
    imread_unicode,
    imwrite_unicode,
    list_images,
    order_points,
    preprocess_basic,
    warp_to_rectangle,
)


POINTS_BY_POSITION = {
    # Left to right: 1=1318, 2=1319, 3=1320, 4=1321.
    # Updated from manual_four_tile_outputs/tile*_rect.json.
    1: np.array(
        [
            [859.6622314453125, 65.7066650390625],
            [1524.9422607421875, 54.75555419921875],
            [1579.69775390625, 1437.3333740234375],
            [90.34666442871094, 1500.30224609375],
        ],
        dtype=np.float32,
    ),
    2: np.array(
        [
            [843.2355346679688, 71.18222045898438],
            [1508.515625, 54.75555419921875],
            [1538.631103515625, 1437.3333740234375],
            [30.115556716918945, 1505.77783203125],
        ],
        dtype=np.float32,
    ),
    3: np.array(
        [
            [835.022216796875, 65.7066650390625],
            [1494.82666015625, 57.49333572387695],
            [1519.4666748046875, 1445.546630859375],
            [16.426666259765625, 1497.564453125],
        ],
        dtype=np.float32,
    ),
    4: np.array(
        [
            [862.4000244140625, 82.13333129882812],
            [1524.9422607421875, 60.23111343383789],
            [1585.17333984375, 1448.284423828125],
            [93.08444213867188, 1505.77783203125],
        ],
        dtype=np.float32,
    ),
}


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent


def position_from_name(path: Path) -> int | None:
    stem = path.stem
    if "1318" in stem:
        return 1
    if "1319" in stem:
        return 2
    if "1320" in stem:
        return 3
    if "1321" in stem:
        return 4
    return None


def process_one_image(
    path: Path,
    points: np.ndarray,
    out_w: int,
    out_h: int,
    out_dir: Path,
    meta_dir: Path,
    position: int,
    do_wb: bool = True,
    do_exposure: bool = True,
    do_glare_reduction: bool = True,
    do_clahe: bool = True,
) -> tuple[Path, Path] | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    bgr = imread_unicode(path)
    if bgr is None:
        return None

    processed = preprocess_basic(
        bgr,
        do_wb=do_wb,
        do_exposure=do_exposure,
        do_glare_reduction=do_glare_reduction,
        do_clahe=do_clahe,
    )
    warped, matrix = warp_to_rectangle(processed, points, out_w=out_w, out_h=out_h)

    out_img = out_dir / f"{path.stem}_rect.png"
    if not imwrite_unicode(out_img, warped):
        raise RuntimeError(f"Failed to write image: {out_img}")

    meta = {
        "src_path": str(path),
        "out_path": str(out_img),
        "position_left_to_right": position,
        "points_raw": points.tolist(),
        "points_ordered": order_points(points).tolist(),
        "H": matrix.tolist(),
        "out_size": [out_w, out_h],
        "mode": "fixed_points_manual_four_tile",
    }
    meta_path = meta_dir / f"{path.stem}_rect.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_img, meta_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess images using the four fixed fix_2 perspective point sets."
    )
    parser.add_argument("--image", type=Path, help="Process a single image.")
    parser.add_argument("--position", type=int, choices=(1, 2, 3, 4), help="Left-to-right position for --image.")
    parser.add_argument("--raw-dir", type=Path, help="Directory containing raw images.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images.")
    parser.add_argument("--out-width", type=int, default=1600)
    parser.add_argument("--out-height", type=int, default=900)
    parser.add_argument("--out-dir", type=Path, help="Directory to save processed images.")
    parser.add_argument("--meta-dir", type=Path, help="Directory to save metadata JSON files.")
    parser.add_argument("--no-wb", action="store_true", help="Disable white balance.")
    parser.add_argument("--no-exposure", action="store_true", help="Disable exposure normalization.")
    parser.add_argument("--no-glare-reduction", action="store_true", help="Disable glare suppression.")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE enhancement.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = resolve_project_root()
    out_dir = args.out_dir or (project_root / "data" / "fix_2_processed")
    meta_dir = args.meta_dir or (project_root / "data" / "fix_2_processed_meta")

    if args.image:
        images = [args.image]
        position = args.position or position_from_name(args.image)
        if position is None:
            parser.error("--image requires --position unless the filename contains 1318/1319/1320/1321.")
        positions = [position]
    else:
        raw_dir = args.raw_dir or (project_root / "data" / "fix_2")
        images = list_images(raw_dir)
        if args.limit is not None:
            images = images[: args.limit]
        if len(images) > 4:
            parser.error("This fixed script supports at most four images. Use --limit 4 or --image.")
        positions = [position_from_name(path) or index for index, path in enumerate(images, start=1)]

    if not images:
        print("No images found to preprocess.")
        return 1

    print(f"Project root: {project_root}")
    print(f"Images to process: {len(images)}")
    print("Position mapping: 1=1318, 2=1319, 3=1320, 4=1321")

    results: list[tuple[Path, Path]] = []
    for image_path, position in zip(images, positions, strict=True):
        try:
            result = process_one_image(
                image_path,
                POINTS_BY_POSITION[position],
                out_w=args.out_width,
                out_h=args.out_height,
                out_dir=out_dir,
                meta_dir=meta_dir,
                position=position,
                do_wb=not args.no_wb,
                do_exposure=not args.no_exposure,
                do_glare_reduction=not args.no_glare_reduction,
                do_clahe=not args.no_clahe,
            )
        except Exception as exc:
            print(f"Skip {image_path.name}: {exc}")
            continue
        if result is not None:
            results.append(result)
            print(f"Saved image: {result[0]}")
            print(f"Saved meta : {result[1]}")

    print(f"Processed {len(results)} / {len(images)} images.")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
