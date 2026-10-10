from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _require_cv2():
    import cv2

    return cv2


def imread_unicode(path: Path):
    cv2 = _require_cv2()
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    cv2 = _require_cv2()
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return True


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_to_rectangle(bgr: np.ndarray, pts4: np.ndarray, out_w: int, out_h: int):
    cv2 = _require_cv2()
    src = order_points(pts4)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped, matrix, src, dst


def fit_preview_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int, float]:
    scale = min(max_width / width, max_height / height, 1.0)
    preview_w = max(1, int(round(width * scale)))
    preview_h = max(1, int(round(height * scale)))
    return preview_w, preview_h, scale


def draw_overlay(preview: np.ndarray, preview_points: list[tuple[int, int]]) -> np.ndarray:
    cv2 = _require_cv2()
    out = preview.copy()
    colors = [(0, 255, 255), (0, 220, 0), (255, 128, 0), (0, 0, 255)]
    labels = ["1", "2", "3", "4"]

    for idx, point in enumerate(preview_points):
        color = colors[idx % len(colors)]
        cv2.circle(out, point, 8, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, point, 11, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            labels[idx],
            (point[0] + 12, point[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    if len(preview_points) >= 2:
        for a, b in zip(preview_points, preview_points[1:]):
            cv2.line(out, a, b, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    if len(preview_points) == 4:
        cv2.line(out, preview_points[-1], preview_points[0], (0, 255, 255), 2, lineType=cv2.LINE_AA)

    help_lines = [
        "Left click 4 board corners. Enter: save | u: undo | r: reset | q/Esc: quit",
        "Click in any order; output JSON stores both raw clicks and ordered TL/TR/BR/BL points.",
    ]
    y = 28
    for line in help_lines:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return out


def collect_points(image: np.ndarray, window_name: str, max_width: int, max_height: int) -> np.ndarray | None:
    cv2 = _require_cv2()
    height, width = image.shape[:2]
    preview_w, preview_h, scale = fit_preview_size(width, height, max_width, max_height)
    preview = cv2.resize(image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    preview_points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(preview_points) < 4:
            preview_points.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, preview_w, preview_h)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            cv2.imshow(window_name, draw_overlay(preview, preview_points))
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                return None
            if key in (ord("u"), 8) and preview_points:
                preview_points.pop()
            if key == ord("r"):
                preview_points.clear()
            if key in (13, 10) and len(preview_points) == 4:
                points = np.array(preview_points, dtype=np.float32) / scale
                return points
    finally:
        cv2.destroyWindow(window_name)


def write_meta(
    meta_path: Path,
    src_path: Path,
    out_img: Path,
    raw_points: np.ndarray,
    ordered_points: np.ndarray,
    matrix: np.ndarray,
    dst_points: np.ndarray,
    out_w: int,
    out_h: int,
    mode: str,
) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "src_path": str(src_path),
        "out_path": str(out_img),
        "points_raw": raw_points.tolist(),
        "points_ordered": ordered_points.tolist(),
        "points_ordered_names": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "dst_points": dst_points.tolist(),
        "H": matrix.tolist(),
        "out_size": [out_w, out_h],
        "mode": mode,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually select 4 perspective points on one image, then save warped image and calibration JSON."
    )
    parser.add_argument("--image", type=Path, required=True, help="Source image to calibrate.")
    parser.add_argument("--out-width", type=int, default=1600)
    parser.add_argument("--out-height", type=int, default=900)
    parser.add_argument("--out-dir", type=Path, default=Path("manual_perspective_outputs"))
    parser.add_argument("--image-name", default=None, help="Output warped image filename. Defaults to <stem>_rect.png.")
    parser.add_argument("--meta-name", default=None, help="Output JSON filename. Defaults to <stem>_rect.json.")
    parser.add_argument("--mode", default="manual_fixed_points", help="Mode string written to JSON.")
    parser.add_argument("--max-preview-width", type=int, default=1400)
    parser.add_argument("--max-preview-height", type=int, default=900)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    src_path = args.image.resolve()
    if not src_path.exists():
        print(f"Image not found: {src_path}")
        return 1
    if src_path.suffix.lower() not in IMAGE_SUFFIXES:
        print(f"Unsupported image type: {src_path.suffix}")
        return 1

    image = imread_unicode(src_path)
    if image is None:
        print(f"Failed to read image: {src_path}")
        return 1

    print("Select 4 board corners in the image window.")
    print("Keys: Enter=save, u=undo, r=reset, q/Esc=quit")
    raw_points = collect_points(
        image,
        window_name="manual_perspective_calibrate",
        max_width=args.max_preview_width,
        max_height=args.max_preview_height,
    )
    if raw_points is None:
        print("Cancelled.")
        return 1

    warped, matrix, ordered_points, dst_points = warp_to_rectangle(
        image,
        raw_points,
        out_w=args.out_width,
        out_h=args.out_height,
    )

    out_dir = args.out_dir.resolve()
    out_img = out_dir / (args.image_name or f"{src_path.stem}_rect.png")
    meta_path = out_dir / (args.meta_name or f"{src_path.stem}_rect.json")

    if not imwrite_unicode(out_img, warped):
        print(f"Failed to write image: {out_img}")
        return 1
    write_meta(
        meta_path=meta_path,
        src_path=src_path,
        out_img=out_img,
        raw_points=raw_points,
        ordered_points=ordered_points,
        matrix=matrix,
        dst_points=dst_points,
        out_w=args.out_width,
        out_h=args.out_height,
        mode=args.mode,
    )

    print(f"Saved warped image: {out_img}")
    print(f"Saved calibration JSON: {meta_path}")
    print("Ordered points for FIXED_POINTS:")
    print(np.array2string(ordered_points, precision=1, suppress_small=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
