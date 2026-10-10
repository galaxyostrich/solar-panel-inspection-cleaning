from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manual_perspective_calibrate import (
    collect_points,
    imread_unicode,
    imwrite_unicode,
    order_points,
    warp_to_rectangle,
)


DEFAULT_IMAGE_NAMES = ["img_1.jpg", "img_2.jpg", "img_3.jpg", "img_4.jpg"]


def save_tile_calibration(
    image_path: Path,
    out_dir: Path,
    tile_index: int,
    raw_points: np.ndarray,
    out_width: int,
    out_height: int,
) -> tuple[Path, Path]:
    image = imread_unicode(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    warped, matrix, ordered_points, dst_points = warp_to_rectangle(
        image,
        raw_points,
        out_w=out_width,
        out_h=out_height,
    )

    tile_name = f"tile{tile_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_out = out_dir / f"{tile_name}_rect.png"
    json_out = out_dir / f"{tile_name}_rect.json"

    if not imwrite_unicode(image_out, warped):
        raise RuntimeError(f"Failed to write image: {image_out}")

    meta = {
        "tile_index": tile_index,
        "src_path": str(image_path),
        "out_path": str(image_out),
        "points_raw": raw_points.tolist(),
        "points_ordered": ordered_points.tolist(),
        "points_ordered_names": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "dst_points": dst_points.tolist(),
        "H": matrix.tolist(),
        "out_size": [out_width, out_height],
        "mode": f"manual_tile{tile_index}",
    }
    json_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return image_out, json_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually calibrate 4 captured images and save one JSON + warped preview per tile."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("."),
        help="Directory containing img_1.jpg, img_2.jpg, img_3.jpg, img_4.jpg.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs=4,
        default=None,
        help="Explicit four image paths, in tile order.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("manual_four_tile_outputs"))
    parser.add_argument("--out-width", type=int, default=1600)
    parser.add_argument("--out-height", type=int, default=900)
    parser.add_argument("--max-preview-width", type=int, default=1400)
    parser.add_argument("--max-preview-height", type=int, default=900)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.images is not None:
        image_paths = [path.resolve() for path in args.images]
    else:
        image_dir = args.image_dir.resolve()
        image_paths = [(image_dir / name).resolve() for name in DEFAULT_IMAGE_NAMES]

    missing = [path for path in image_paths if not path.exists()]
    if missing:
        print("Missing image(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    out_dir = args.out_dir.resolve()
    print("Four-tile manual calibration")
    print("Controls: left click 4 board corners, Enter=save tile, u=undo, r=reset, q/Esc=quit")
    print("Click order can be any order; output stores ordered TL/TR/BR/BL points.")

    results: list[tuple[Path, Path]] = []
    for tile_index, image_path in enumerate(image_paths, start=1):
        image = imread_unicode(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return 1

        print(f"\nTile {tile_index}/4: {image_path}")
        raw_points = collect_points(
            image,
            window_name=f"calibrate_tile{tile_index}_{image_path.name}",
            max_width=args.max_preview_width,
            max_height=args.max_preview_height,
        )
        if raw_points is None:
            print("Cancelled.")
            return 1

        image_out, json_out = save_tile_calibration(
            image_path=image_path,
            out_dir=out_dir,
            tile_index=tile_index,
            raw_points=raw_points,
            out_width=args.out_width,
            out_height=args.out_height,
        )
        results.append((image_out, json_out))
        print(f"Saved warped image: {image_out}")
        print(f"Saved calibration JSON: {json_out}")
        print("Ordered FIXED_POINTS:")
        print(np.array2string(order_points(raw_points), precision=1, suppress_small=True))

    print("\nDone. Calibration outputs:")
    for image_out, json_out in results:
        print(f"  {image_out}")
        print(f"  {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
