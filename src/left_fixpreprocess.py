from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FIXED_POINTS = np.array(
    [
        [820.0, 250.0],
        [2239.0, 184.0],
        [3246.0, 1899.0],
        [2.0, 2125.0],
    ],
    dtype=np.float32,
)


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
    encoded.tofile(str(path))
    return True


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    img = bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    mean = (b_mean + g_mean + r_mean) / 3.0 + 1e-6
    b *= mean / (b_mean + 1e-6)
    g *= mean / (g_mean + 1e-6)
    r *= mean / (r_mean + 1e-6)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_l_channel(bgr: np.ndarray, clip: float = 2.0, tile: tuple[int, int] = (8, 8)) -> np.ndarray:
    cv2 = _require_cv2()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tile)
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)


def normalize_exposure(bgr: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gain = target_mean / (float(gray.mean()) + 1e-6)
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def glare_mask(bgr: np.ndarray, sat_thr: int = 30, val_thr: int = 240) -> np.ndarray:
    cv2 = _require_cv2()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    mask = ((value >= val_thr) & (saturation <= sat_thr)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask


def reduce_glare_exposure(
    bgr: np.ndarray,
    sat_thr: int = 40,
    val_thr: int = 225,
    exposure_scale: float = 0.55,
    blur_ksize: int = 21,
) -> np.ndarray:
    cv2 = _require_cv2()
    mask = glare_mask(bgr, sat_thr=sat_thr, val_thr=val_thr)
    if not np.any(mask):
        return bgr.copy()

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    alpha = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    original_v = hsv[:, :, 2]
    reduced_v = original_v * exposure_scale
    hsv[:, :, 2] = original_v * (1.0 - alpha) + reduced_v * alpha
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def preprocess_basic(
    bgr: np.ndarray,
    do_wb: bool = True,
    do_exposure: bool = True,
    do_glare_reduction: bool = True,
    do_clahe: bool = True,
) -> np.ndarray:
    out = bgr.copy()
    if do_wb:
        out = gray_world_white_balance(out)
    if do_exposure:
        out = normalize_exposure(out, target_mean=130.0)
    if do_glare_reduction:
        out = reduce_glare_exposure(out, sat_thr=40, val_thr=225, exposure_scale=0.55, blur_ksize=21)
    if do_clahe:
        out = clahe_l_channel(out, clip=2.0, tile=(8, 8))
    return out


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_to_rectangle(bgr: np.ndarray, pts4: np.ndarray, out_w: int = 1600, out_h: int = 900):
    cv2 = _require_cv2()
    src = order_points(pts4)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped, matrix


def process_one_image(
    path: Path,
    out_w: int = 1600,
    out_h: int = 900,
    out_dir: Path | None = None,
    meta_dir: Path | None = None,
    do_wb: bool = True,
    do_exposure: bool = True,
    do_glare_reduction: bool = True,
    do_clahe: bool = True,
) -> tuple[Path, Path] | None:
    base_dir = Path.cwd().resolve()
    proc_dir = out_dir or (base_dir / "processed")
    meta_dir = meta_dir or (base_dir / "processed_meta")
    proc_dir.mkdir(parents=True, exist_ok=True)
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
    warped, matrix = warp_to_rectangle(processed, FIXED_POINTS, out_w=out_w, out_h=out_h)

    out_img = proc_dir / f"{path.stem}_rect.png"
    if not imwrite_unicode(out_img, warped):
        raise RuntimeError(f"Failed to write image: {out_img}")

    meta = {
        "src_path": str(path),
        "out_path": str(out_img),
        "points_raw": FIXED_POINTS.tolist(),
        "points_ordered": order_points(FIXED_POINTS).tolist(),
        "H": matrix.tolist(),
        "out_size": [out_w, out_h],
        "mode": "fixed_points_left",
    }
    meta_path = meta_dir / f"{path.stem}_rect.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_img, meta_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess images using fixed LEFT perspective points.")
    parser.add_argument("--image", type=Path, help="Process a single image.")
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

    if args.image:
        images = [args.image]
    else:
        raw_dir = args.raw_dir or Path.cwd()
        images = list_images(raw_dir)
        if args.limit is not None:
            images = images[: args.limit]

    if not images:
        print("No images found to preprocess.")
        return 1

    print(f"Working dir: {Path.cwd().resolve()}")
    print(f"Images to process: {len(images)}")
    print(f"Fixed points: {order_points(FIXED_POINTS).tolist()}")

    results: list[tuple[Path, Path]] = []
    for image_path in images:
        try:
            result = process_one_image(
                image_path,
                out_w=args.out_width,
                out_h=args.out_height,
                out_dir=args.out_dir,
                meta_dir=args.meta_dir,
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
