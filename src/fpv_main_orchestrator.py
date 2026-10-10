from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


# ================================================================
# Adjustable defaults
# ================================================================
DEFAULT_PORT = 5001
DEFAULT_CAPTURE_DIR = Path.home() / "fpv_captures"
DEFAULT_WORK_DIR = Path.home() / "fpv_work"
DEFAULT_CAPTURE_X_CMS = [9.6, 17.6, 25.5, 33.2]
DEFAULT_IMAGE_NAMES = ["img_1.jpg", "img_2.jpg", "img_3.jpg", "img_4.jpg"]
DEFAULT_CLEAN_X_IGNORE_BELOW_CM = 3.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================
# Framed socket helpers
# ================================================================

def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.extend(chunk)
    return bytes(chunks)


def send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def recv_frame(sock: socket.socket) -> bytes:
    header = recv_exact(sock, 8)
    (size,) = struct.unpack("!Q", header)
    return recv_exact(sock, size)


def send_json(sock: socket.socket, obj: dict[str, Any]) -> None:
    send_frame(sock, json.dumps(obj).encode("utf-8"))


def recv_json(sock: socket.socket) -> dict[str, Any]:
    return json.loads(recv_frame(sock).decode("utf-8"))


def send_file(sock: socket.socket, path: Path) -> None:
    send_json(sock, {"name": path.name, "size": path.stat().st_size})
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            send_frame(sock, chunk)
    send_frame(sock, b"")


def recv_file(sock: socket.socket, out_dir: Path) -> Path:
    meta = recv_json(sock)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / meta["name"]
    with out_path.open("wb") as f:
        while True:
            chunk = recv_frame(sock)
            if not chunk:
                break
            f.write(chunk)
    return out_path


# ================================================================
# Camera capture on Raspberry Pi
# ================================================================

def run_cmd(cmd: list[str]) -> None:
    log("RUN: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture_with_rpicam(output_path: Path, width: int, height: int, timeout_ms: int) -> None:
    cmd = [
        "rpicam-still",
        "-o", str(output_path),
        "--width", str(width),
        "--height", str(height),
        "--timeout", str(timeout_ms),
        "--nopreview",
    ]
    run_cmd(cmd)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Camera command finished but image was not created: {output_path}")
    log(f"Captured {output_path} ({output_path.stat().st_size} bytes)")


def capture_images(
    capture_dir: Path,
    width: int,
    height: int,
    timeout_ms: int,
    pause_s: float,
    image_names: list[str],
    capture_x_cms: list[float],
    shuo_script: Path | None = None,
    rpm_x: float = 20.0,
    after_capture: Callable[[list[Path]], None] | None = None,
) -> list[Path]:
    if len(image_names) != len(capture_x_cms):
        raise ValueError("image_names and capture_x_cms must have the same length")
    if not image_names:
        raise ValueError("At least one capture position is required")

    capture_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [capture_dir / name for name in image_names]

    shuo = import_module_from_path(shuo_script, "shuo_capture") if shuo_script is not None else None
    capture_complete = False
    after_capture_exc: Exception | None = None
    try:
        if shuo is not None:
            shuo.init_gpio()
            if hasattr(shuo, "set_clean_servo_up"):
                shuo.set_clean_servo_up()

        for idx, (x_cm, image_path) in enumerate(zip(capture_x_cms, image_paths), start=1):
            if shuo is not None:
                log(f"Moving X to {x_cm:.2f} cm for image {idx}/{len(image_paths)}")
                shuo.move_x_to(float(x_cm), rpm_x=rpm_x)
                time.sleep(pause_s)

            capture_with_rpicam(image_path, width=width, height=height, timeout_ms=timeout_ms)
        capture_complete = True
    finally:
        if capture_complete and after_capture is not None:
            try:
                after_capture(image_paths)
            except Exception as exc:
                after_capture_exc = exc
                log(f"After-capture callback warning: {exc}")

        if shuo is not None:
            try:
                log("Returning X to 0 after capture")
                shuo.move_x_to(0.0, rpm_x=rpm_x)
                if hasattr(shuo, "set_clean_servo_up"):
                    shuo.set_clean_servo_up()
                if hasattr(shuo, "release_servo_pwm"):
                    shuo.release_servo_pwm()
                shuo.disable_driver()
                shuo._import_gpio().cleanup()
                shuo._gpio_ready = False
            except Exception as exc:
                log(f"Capture cleanup warning: {exc}")

        if after_capture_exc is not None:
            raise after_capture_exc

    return image_paths


# ================================================================
# Laptop processing
# ================================================================

def import_module_from_path(path_or_dir: Path, module_name: str):
    import importlib.util

    path_or_dir = path_or_dir.resolve()
    if path_or_dir.is_dir():
        file_path = path_or_dir / f"{module_name}.py"
    else:
        file_path = path_or_dir

    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find module file: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def process_images_on_laptop(
    input_dir: Path,
    work_dir: Path,
    model_path: Path,
    detector_script: Path,
    shuo_script: Path,
    conf: float,
    iou: float,
    out_width: int,
    out_height: int,
) -> dict[str, Any]:
    detector = import_module_from_path(detector_script, "run_pair_preprocess_detect_main")
    shuo = import_module_from_path(shuo_script, "shuo_main")

    output_paths = detector.run_pair(
        input_dir=input_dir,
        model_path=model_path,
        out_dir=work_dir / "outputs",
        conf=conf,
        iou=iou,
        out_width=out_width,
        out_height=out_height,
    )

    label_paths: list[Path] = []
    overlay_paths: list[Path] = []
    for item in output_paths:
        if isinstance(item, tuple) and len(item) == 2:
            label_path, overlay_path = item
            label_paths.append(Path(label_path))
            overlay_paths.append(Path(overlay_path))
        else:
            label_paths.append(Path(item))

    if not label_paths:
        raise RuntimeError("No label files were produced by detection script")

    # ---- Step 1: load raw spots (per-tile YOLO detections -> board cm) ----
    spots = []
    for image_index, label_path in enumerate(label_paths):
        spots += shuo.load_label_file(Path(label_path), image_index=image_index)

    log(f"Loaded {len(spots)} raw detection(s) across {len(label_paths)} tile(s)")

    # ---- Step 2: greedy disk-pack into cleaning targets ----
    merged = shuo.disk_pack_spots(spots)
    ordered = shuo.sort_targets_by_x(merged)

    log(f"Disk-pack: {len(spots)} raw detections -> {len(ordered)} cleaning target(s)")
    for i, t in enumerate(ordered, start=1):
        log(
            f"  T{i}: center=({t.board_x_cm:.2f},{t.board_y_cm:.2f}) cm, "
            f"span=({t.span_x_cm:.2f},{t.span_y_cm:.2f}) cm, "
            f"merged_from={len(t.source_spot_indices)} spot(s)"
        )

    # ---- Step 3: merge-plan overlays (one per tile), drawn on top of YOLO overlays ----
    merge_overlay_paths: list[str] = []
    merge_overlay_dir = (work_dir / "outputs" / "merge_overlays").resolve()
    merge_overlay_dir.mkdir(parents=True, exist_ok=True)
    for tile_idx, yolo_overlay in enumerate(overlay_paths):
        out_name = f"{yolo_overlay.stem}_merged{yolo_overlay.suffix or '.png'}"
        out_path = merge_overlay_dir / out_name
        try:
            shuo.draw_merge_overlay(
                base_image_path=yolo_overlay,
                spots=spots,
                targets=ordered,
                image_index=tile_idx,
                out_path=out_path,
            )
            merge_overlay_paths.append(str(out_path))
            log(f"  Merge overlay (tile {tile_idx}): {out_path}")
        except Exception as exc:
            log(f"  Merge overlay failed for tile {tile_idx}: {exc}")

    # ---- Step 4: build JSON response ----
    targets = []
    ignored_targets = 0
    for t in ordered:
        raw_stepper_x = float(t.stepper_x_cm)
        if raw_stepper_x <= DEFAULT_CLEAN_X_IGNORE_BELOW_CM:
            ignored_targets += 1
            continue
        shifted_stepper_x = raw_stepper_x - DEFAULT_CLEAN_X_IGNORE_BELOW_CM
        targets.append(
            {
                "index": len(targets) + 1,
                "board_x_cm": round(float(t.board_x_cm), 3),
                "board_y_cm": round(float(t.board_y_cm), 3),
                "stepper_x_cm": round(shifted_stepper_x, 3),
                "raw_stepper_x_cm": round(raw_stepper_x, 3),
                "stepper_y_cm": round(float(t.stepper_y_cm), 3),
                "span_x_cm": round(float(t.span_x_cm), 3),
                "span_y_cm": round(float(t.span_y_cm), 3),
                "bbox_cm": [round(float(v), 3) for v in t.bbox_cm],
                "num_source_spots": len(t.source_spot_indices),
                "source_spot_indices": list(t.source_spot_indices),
            }
        )
    log(
        f"Clean X remap: ignored {ignored_targets} target(s) at raw X <= "
        f"{DEFAULT_CLEAN_X_IGNORE_BELOW_CM:.2f} cm; shifted remaining X by "
        f"-{DEFAULT_CLEAN_X_IGNORE_BELOW_CM:.2f} cm"
    )

    return {
        "targets": targets,
        "label_files": [str(p) for p in label_paths],
        "overlay_files": [str(p) for p in overlay_paths],
        "merge_overlay_files": merge_overlay_paths,
        "num_raw_spots": len(spots),
        "num_ignored_targets": ignored_targets,
        "clean_x_ignore_below_cm": DEFAULT_CLEAN_X_IGNORE_BELOW_CM,
        "num_targets": len(targets),
    }


# ================================================================
# Raspberry Pi motor execution
# ================================================================

def _group_target_dicts_by_column(
    targets: list[dict[str, Any]],
    tolerance_cm: float,
) -> list[list[dict[str, Any]]]:
    """Mirror of shuo.group_targets_by_column for flat JSON target dicts.

    Two targets share a column iff their stepper X differ by at most
    `tolerance_cm`. Columns are returned left-to-right. Within each
    column, targets are sorted by ASCENDING stepper Y so smaller Y
    positions are cleaned before larger Y positions at the same X.
    """
    if not targets:
        return []

    by_x = sorted(
        targets,
        key=lambda t: (float(t["stepper_x_cm"]), float(t["stepper_y_cm"])),
    )
    columns: list[list[dict[str, Any]]] = []
    seed_x: list[float] = []

    for t in by_x:
        tx = float(t["stepper_x_cm"])
        if columns and abs(tx - seed_x[-1]) <= tolerance_cm:
            columns[-1].append(t)
        else:
            columns.append([t])
            seed_x.append(tx)

    for col in columns:
        col.sort(key=lambda t: float(t["stepper_y_cm"]))

    return columns


def apply_clean_x_offset(x_cm: float, large_offset_cm: float = 2.0, small_offset_cm: float = 0.5) -> float:
    if large_offset_cm <= 0:
        return x_cm
    if x_cm > large_offset_cm:
        return max(0.0, x_cm - large_offset_cm)
    if x_cm >= 1.0:
        return max(0.0, x_cm - small_offset_cm)
    return x_cm


def execute_targets_on_pi(
    targets: list[dict[str, Any]],
    shuo_script: Path,
    rpm_x: float,
    rpm_y: float,
    dwell_s: float,
    return_home: bool,
    clean_x_offset_cm: float,
    x_column_tolerance_cm: float | None = None,
) -> None:
    shuo = import_module_from_path(shuo_script, "shuo_motor")

    tol = (
        float(x_column_tolerance_cm)
        if x_column_tolerance_cm is not None
        else float(getattr(shuo, "X_COLUMN_TOLERANCE_CM", 0.3))
    )

    columns = _group_target_dicts_by_column(targets, tolerance_cm=tol)
    total = sum(len(c) for c in columns)
    log(
        f"Column-chaining: {len(targets)} target(s) -> {len(columns)} column(s) "
        f"(tolerance={tol:.2f} cm)"
    )
    for c_idx, col in enumerate(columns, start=1):
        col_x = sum(float(t["stepper_x_cm"]) for t in col) / len(col)
        y_list = ", ".join(f"{float(t['stepper_y_cm']):.2f}" for t in col)
        log(f"  Col {c_idx} @ x={col_x:.2f} cm  -> Y order (smallest first): [{y_list}]")

    shuo.init_gpio()
    GPIO = shuo._import_gpio()
    if hasattr(shuo, "set_clean_servo_up"):
        log("Setting clean servo up")
        shuo.set_clean_servo_up()
    try:
        visited = 0
        for c_idx, col in enumerate(columns, start=1):
            col_x = sum(float(t["stepper_x_cm"]) for t in col) / len(col)
            clean_x = apply_clean_x_offset(col_x, large_offset_cm=clean_x_offset_cm)
            log(f"=== Column {c_idx}/{len(columns)} @ X={col_x:.2f} cm "
                f"({len(col)} target(s)) ===")
            if clean_x != col_x:
                log(f"Applying clean X offset: target X {col_x:.2f} cm -> move X {clean_x:.2f} cm")
            else:
                log(f"Moving X to {clean_x:.2f} cm (column position)")
            shuo.move_x_to(clean_x, rpm_x=rpm_x)

            for target in col:
                visited += 1
                y_cm = float(target["stepper_y_cm"])
                idx = target.get("index", "?")

                log(f"[{visited}/{total}] Target {idx}: Y={y_cm:.2f} cm "
                    f"(X held at {clean_x:.2f} cm; raw target X {col_x:.2f} cm)")
                shuo.move_y_to(y_cm, rpm_y=rpm_y)

                log(f"Dwelling/cleaning for {dwell_s:.1f} s")
                try:
                    if hasattr(shuo, "DC_MOTOR_PIN"):
                        if hasattr(shuo, "set_clean_servo_down"):
                            log("Clean servo down")
                            shuo.set_clean_servo_down()
                        log("DC motor on")
                        GPIO.output(shuo.DC_MOTOR_PIN, GPIO.HIGH)
                        time.sleep(dwell_s)
                        GPIO.output(shuo.DC_MOTOR_PIN, GPIO.LOW)
                        log("DC motor off")
                    else:
                        time.sleep(dwell_s)
                finally:
                    if hasattr(shuo, "DC_MOTOR_PIN"):
                        GPIO.output(shuo.DC_MOTOR_PIN, GPIO.LOW)
                    if hasattr(shuo, "set_clean_servo_up"):
                        log("Clean servo up")
                        shuo.set_clean_servo_up()

            # Y returns to 0 only once per column, after all its Y targets
            # have been visited from smaller Y to larger Y.
            log(f"End of column {c_idx}: returning Y to 0")
            shuo.move_y_to(0.0, rpm_y=rpm_y)

        if return_home:
            log("Final return: ensuring Y is 0 before returning X")
            shuo.move_y_to(0.0, rpm_y=rpm_y)
            log("Returning X to 0")
            shuo.move_x_to(0.0, rpm_x=rpm_x)
    finally:
        try:
            if hasattr(shuo, "set_clean_servo_up"):
                shuo.set_clean_servo_up()
            if hasattr(shuo, "release_servo_pwm"):
                shuo.release_servo_pwm()
            shuo.disable_driver()
            shuo._import_gpio().cleanup()
        except Exception:
            pass


# ================================================================
# Role: Raspberry Pi
# ================================================================

def send_images_to_laptop(
    images: list[Path],
    laptop_host: str,
    port: int,
    capture_x_cms: list[float],
) -> dict[str, Any]:
    log(f"Connecting to laptop at {laptop_host}:{port}")
    with socket.create_connection((laptop_host, port), timeout=30) as sock:
        send_json(sock, {
            "type": "capture_batch",
            "project": "fpv_solar_cleaning",
            "image_count": len(images),
            "capture_x_cms": capture_x_cms,
            "timestamp": time.time(),
        })
        ack = recv_json(sock)
        if ack.get("status") != "ready":
            raise RuntimeError(f"Laptop not ready: {ack}")

        for image_path in images:
            send_file(sock, image_path)

        result = recv_json(sock)
        if result.get("status") != "ok":
            raise RuntimeError(f"Laptop returned error: {result}")
        return result


def run_pi_role(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).resolve()
    shuo_script = Path(args.shuo_script).resolve()
    capture_x_cms = [float(x) for x in args.capture_x_cms]
    image_names = list(args.image_names)

    log(f"Capturing {len(capture_x_cms)} images into {capture_dir}")
    laptop_future: Future[dict[str, Any]] | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        def start_laptop_processing(captured_images: list[Path]) -> None:
            nonlocal laptop_future
            log("Starting laptop transfer while X returns to 0")
            laptop_future = executor.submit(
                send_images_to_laptop,
                captured_images,
                args.laptop_host,
                args.port,
                capture_x_cms,
            )

        images = capture_images(
            capture_dir=capture_dir,
            width=args.width,
            height=args.height,
            timeout_ms=args.capture_timeout_ms,
            pause_s=args.pause_between_shots_s,
            image_names=image_names,
            capture_x_cms=capture_x_cms,
            shuo_script=shuo_script,
            rpm_x=args.rpm_x,
            after_capture=start_laptop_processing,
        )

        if laptop_future is None:
            log("Starting laptop transfer after capture")
            laptop_future = executor.submit(
                send_images_to_laptop,
                images,
                args.laptop_host,
                args.port,
                capture_x_cms,
            )

        log("Waiting for laptop processing result")
        result = laptop_future.result()

    targets = result.get("targets", [])
    num_raw = result.get("num_raw_spots", "?")
    log(f"Laptop returned {len(targets)} cleaning target(s) merged from {num_raw} raw detection(s)")
    for t in targets:
        raw_x = t.get("raw_stepper_x_cm", t["stepper_x_cm"])
        log(
            f"Target {t['index']}: board=({t['board_x_cm']:.2f}, {t['board_y_cm']:.2f}) cm, "
            f"stepper=({t['stepper_x_cm']:.2f}, {t['stepper_y_cm']:.2f}) cm, "
            f"raw_x={float(raw_x):.2f} cm, "
            f"merged_from={t.get('num_source_spots', '?')} spot(s)"
        )

    if args.dry_run:
        log("Dry run enabled: no motor movement executed")
        return 0

    if not targets:
        log("No salt residue detected. Nothing to clean.")
        return 0

    execute_targets_on_pi(
        targets=targets,
        shuo_script=shuo_script,
        rpm_x=args.rpm_x,
        rpm_y=args.rpm_y,
        dwell_s=args.clean_dwell_s,
        return_home=not args.no_return_home,
        clean_x_offset_cm=args.clean_x_offset_cm,
        x_column_tolerance_cm=args.x_column_tolerance_cm,
    )
    return 0


# ================================================================
# Role: Laptop server
# ================================================================

def run_laptop_role(args: argparse.Namespace) -> int:
    host = args.bind_host
    port = args.port
    base_work_dir = Path(args.work_dir).resolve()
    detector_script = Path(args.detector_script).resolve()
    shuo_script = Path(args.shuo_script).resolve()
    model_path = Path(args.model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    base_work_dir.mkdir(parents=True, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        log(f"Laptop server listening on {host}:{port}")

        conn, addr = server.accept()
        with conn:
            log(f"Connected by {addr}")
            hello = recv_json(conn)
            if hello.get("type") != "capture_batch":
                send_json(conn, {"status": "error", "message": "Expected capture_batch"})
                return 1
            image_count = int(hello.get("image_count", 2))
            if image_count <= 0:
                send_json(conn, {"status": "error", "message": "image_count must be positive"})
                return 1

            send_json(conn, {"status": "ready"})

            session_name = time.strftime("session_%Y%m%d_%H%M%S")
            session_dir = base_work_dir / session_name
            input_dir = session_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)

            received_images = [recv_file(conn, input_dir) for _ in range(image_count)]
            log("Received images: " + ", ".join(p.name for p in received_images))

            try:
                result = process_images_on_laptop(
                    input_dir=input_dir,
                    work_dir=session_dir,
                    model_path=model_path,
                    detector_script=detector_script,
                    shuo_script=shuo_script,
                    conf=args.conf,
                    iou=args.iou,
                    out_width=args.out_width,
                    out_height=args.out_height,
                )
                result["status"] = "ok"
                result["session_dir"] = str(session_dir)
                send_json(conn, result)
                log(f"Sent back {result['num_targets']} target(s)")
            except Exception as exc:
                send_json(conn, {"status": "error", "message": str(exc)})
                log(f"Processing failed: {exc}")
                return 1

    return 0


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single orchestrator file for FPV image capture, laptop detection, coordinate return, and Pi motor control."
    )
    subparsers = parser.add_subparsers(dest="role", required=True)

    pi = subparsers.add_parser("pi", help="Run on Raspberry Pi")
    pi.add_argument("--laptop-host", required=True, help="Laptop IP address")
    pi.add_argument("--port", type=int, default=DEFAULT_PORT)
    pi.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR))
    pi.add_argument("--width", type=int, default=3280)
    pi.add_argument("--height", type=int, default=2464)
    pi.add_argument("--capture-timeout-ms", type=int, default=1200)
    pi.add_argument("--pause-between-shots-s", type=float, default=0.8)
    pi.add_argument("--capture-x-cms", type=float, nargs="+", default=DEFAULT_CAPTURE_X_CMS)
    pi.add_argument("--image-names", nargs="+", default=DEFAULT_IMAGE_NAMES)
    pi.add_argument("--rpm-x", type=float, default=20.0)
    pi.add_argument("--rpm-y", type=float, default=20.0)
    pi.add_argument("--clean-dwell-s", type=float, default=1.0)
    pi.add_argument(
        "--clean-x-offset-cm",
        type=float,
        default=0.0,
        help="Legacy extra cleaning X offset. Keep 0 unless you intentionally want an additional Pi-side offset.",
    )
    pi.add_argument("--x-column-tolerance-cm", type=float, default=0.3,
                    help="Targets whose stepper X differs by <= this value are "
                         "chained into one column and cleaned from smaller Y "
                         "to larger Y.")
    pi.add_argument("--no-return-home", action="store_true")
    pi.add_argument("--dry-run", action="store_true")
    pi.add_argument(
        "--shuo-script",
        default="./shuo.py",
        help="Path to your stepper + coordinate script on the Pi",
    )

    laptop = subparsers.add_parser("laptop", help="Run on laptop as image-processing server")
    laptop.add_argument("--bind-host", default="0.0.0.0")
    laptop.add_argument("--port", type=int, default=DEFAULT_PORT)
    laptop.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    laptop.add_argument("--conf", type=float, default=0.1)
    laptop.add_argument("--iou", type=float, default=0.5)
    laptop.add_argument("--out-width", type=int, default=1600)
    laptop.add_argument("--out-height", type=int, default=900)
    laptop.add_argument("--model-path", default="./best.pt")
    laptop.add_argument(
        "--detector-script",
        default="./run_pair_preprocess_detect.py",
        help="Path to your preprocessing + YOLO script on the laptop",
    )
    laptop.add_argument(
        "--shuo-script",
        default="./shuo.py",
        help="Path to shuo.py on the laptop for label->coordinate conversion",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.role == "pi":
            return run_pi_role(args)
        if args.role == "laptop":
            return run_laptop_role(args)
        parser.error("Unknown role")
        return 2
    except KeyboardInterrupt:
        log("Interrupted by user")
        return 130
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
