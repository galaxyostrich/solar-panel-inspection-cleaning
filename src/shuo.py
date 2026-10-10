"""
shuo.py  --  combined stepper driver + auto-clean pipeline.

Two ways to use this file:

  1) As a library (import its motor functions from another script):
        import shuo
        shuo.move_to_xy(8.0, 5.0)

  2) As an executable that reads YOLO label files for the two rectified
     images and drives the stepper to every detection:
        python shuo.py --label-left img1.txt --label-right img2.txt
        python shuo.py --label-left img1.txt --label-right img2.txt --dry-run

YOLO label line format (one detection per line):
        class_id   xc   yc   w   h        (xc/yc/w/h all normalized in [0,1])

Merging behavior:
    Before driving the motor, raw YOLO detections are collapsed into
    "cleaning targets" via 2D greedy disk packing. A single target covers
    one sponge placement (SPONGE_EFFECTIVE_CM in both X and Y).  Adjacent
    or overlapping detections that collectively fit within a 3.8 cm square
    are merged into a single target.  Larger residues produce multiple
    adjacent targets that jointly cover the entire detected region.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

# RPi.GPIO is only present on the Pi. Importing it lazily lets --dry-run
# work on a laptop.
def _import_gpio():
    import RPi.GPIO as GPIO
    return GPIO


# =====================================================================
# CALIBRATION  --  edit to match your rig
# =====================================================================

# Physical area each rectified image covers, in cm.
# The panel is 30.8 cm wide with 16 columns. The current four-image
# capture mode maps each rectified tile to 4 columns, so each tile spans
# one quarter of the panel width.
PANEL_WIDTH_CM = 30.8
IMAGE_WIDTH_CM  = PANEL_WIDTH_CM / 4.0
IMAGE_HEIGHT_CM = 21.1

# Top-left corner of each rectified image expressed in board cm.
# Four captures map to columns 1-4, 5-8, 9-12, and 13-16.
IMAGE_ORIGINS_CM = [
    (0.0 * IMAGE_WIDTH_CM, 0.0),   # image 1, columns 1-4
    (1.0 * IMAGE_WIDTH_CM, 0.0),   # image 2, columns 5-8
    (2.0 * IMAGE_WIDTH_CM, 0.0),   # image 3, columns 9-12
    (3.0 * IMAGE_WIDTH_CM, 0.0),   # image 4, columns 13-16
]

# Offset between "board (0,0)" and "stepper command (0,0)".
# Measure once: jog the nozzle to a known board point, read the
# stepper cm, subtract the board cm.
CAM_OFFSET_X_CM = 0.0   # TODO
CAM_OFFSET_Y_CM = 1.8   # Add to all txt-derived Y targets before moving the brush.

# Stepper feed rates used by the auto-clean main()
RPM_X = 20
RPM_Y = 20


# =====================================================================
# Sponge / disk-pack parameters
# =====================================================================
# Physical diameter of the cleaning sponge in cm. Used for drawing the
# sponge footprint in the merge overlay.
SPONGE_DIAMETER_CM = 4.0

# Effective sponge reach used by disk packing.  Slightly smaller than
# SPONGE_DIAMETER_CM to give a safety margin against coordinate mapping
# error and edge softness of the sponge. Two detections merge into a
# single cleaning target iff their combined bounding box in board-cm
# space fits within SPONGE_EFFECTIVE_CM in BOTH X and Y.
SPONGE_EFFECTIVE_CM = 3.8

# Column-chaining tolerance for the cleaning sequence.
# Two cleaning targets whose stepper X coordinates differ by at most this
# value are treated as belonging to the same "X column": they are visited
# from smaller Y to larger Y within one column. Y returns to 0 once at
# the end of each column, before moving to the next X.
X_COLUMN_TOLERANCE_CM = 0.3


# =====================================================================
# GPIO Definitions
# =====================================================================
X_STEP_PIN = 18
X_DIR_PIN  = 23
X_EN_PIN   = 24

Y_STEP_PIN = 19
Y_DIR_PIN  = 20
Y_EN_PIN   = 21

STOP_PIN = 16   # emergency stop button
SERVO_PIN = 13
DC_MOTOR_PIN = 26

# ===== Motor Parameters =====
FULL_STEPS_PER_REV = 200
X_MICROSTEP = 4
Y_MICROSTEP = 16
X_STEPS_PER_REV = FULL_STEPS_PER_REV * X_MICROSTEP
Y_STEPS_PER_REV = FULL_STEPS_PER_REV * Y_MICROSTEP

STEP_TO_CM_X = 0.02
STEP_TO_CM_Y = 0.02
CM_TO_STEP_X = 1 / STEP_TO_CM_X
CM_TO_STEP_Y = 1 / STEP_TO_CM_Y

# Flip this if positive Y movement is physically reversed.
Y_DIR_INVERT = False

# GPIO13 controls the DC motor direction/position servo.
# Keep it vertical by default; lower it only while the DC motor is cleaning.
CLEAN_SERVO_UP_ANGLE = 0.0
CLEAN_SERVO_DOWN_ANGLE = -80.0

current_x_cm = 0.0
current_y_cm = 0.0

_gpio_ready = False
servo_pwm = None

def release_servo_pwm() -> None:
    global servo_pwm
    if servo_pwm is not None:
        try:
            servo_pwm.stop()
        except Exception:
            pass
        servo_pwm = None


def ensure_servo_pwm():
    global servo_pwm
    if servo_pwm is None:
        GPIO = _import_gpio()
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        servo_pwm = GPIO.PWM(SERVO_PIN, 50)
        servo_pwm.start(0)
    return servo_pwm

def init_gpio() -> None:
    global _gpio_ready, servo_pwm
    if _gpio_ready:
        return
    GPIO = _import_gpio()
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(X_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(X_DIR_PIN,  GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(X_EN_PIN,   GPIO.OUT, initial=GPIO.HIGH)

    GPIO.setup(Y_STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(Y_DIR_PIN,  GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(Y_EN_PIN,   GPIO.OUT, initial=GPIO.HIGH)

    GPIO.setup(STOP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.setup(DC_MOTOR_PIN, GPIO.OUT, initial=GPIO.LOW)
    _gpio_ready = True


# =====================================================================
# Motor primitives
# =====================================================================
def enable_driver():
    GPIO = _import_gpio()
    GPIO.output(X_EN_PIN, GPIO.LOW)
    GPIO.output(Y_EN_PIN, GPIO.LOW)
    time.sleep(0.002)


def disable_driver():
    GPIO = _import_gpio()
    GPIO.output(X_EN_PIN, GPIO.HIGH)
    GPIO.output(Y_EN_PIN, GPIO.HIGH)


def emergency_stop() -> bool:
    GPIO = _import_gpio()
    if GPIO.input(STOP_PIN) == GPIO.LOW:
        print("[EMERGENCY STOP]")
        disable_driver()
        return True
    return False


def step_pulses(step_x: bool, step_y: bool, delay_s: float) -> None:
    GPIO = _import_gpio()
    if step_x:
        GPIO.output(X_STEP_PIN, GPIO.HIGH)
    if step_y:
        GPIO.output(Y_STEP_PIN, GPIO.HIGH)
    time.sleep(delay_s)
    if step_x:
        GPIO.output(X_STEP_PIN, GPIO.LOW)
    if step_y:
        GPIO.output(Y_STEP_PIN, GPIO.LOW)
    time.sleep(delay_s)


def move_xy_steps(steps_x: int, steps_y: int, rpm_x: float = 60, rpm_y: float = 60,
                  dir_x: int = 1, dir_y: int = 1) -> None:
    if steps_x <= 0 and steps_y <= 0:
        return

    init_gpio()
    GPIO = _import_gpio()
    enable_driver()

    GPIO.output(X_DIR_PIN, GPIO.HIGH if dir_x else GPIO.LOW)
    y_dir_level = GPIO.HIGH if dir_y else GPIO.LOW
    if Y_DIR_INVERT:
        y_dir_level = GPIO.LOW if dir_y else GPIO.HIGH
    GPIO.output(Y_DIR_PIN, y_dir_level)
    time.sleep(0.001)

    pps_x = rpm_x * X_STEPS_PER_REV / 60.0 if steps_x > 0 and rpm_x > 0 else 0.0
    pps_y = rpm_y * Y_STEPS_PER_REV / 60.0 if steps_y > 0 and rpm_y > 0 else 0.0
    if pps_x <= 0 and pps_y <= 0:
        return

    max_pps = max(pps_x, pps_y)
    period = 1.0 / max_pps
    half_period = max(50e-6, period / 2)

    ratio_x = pps_x / max_pps if pps_x > 0 else 0.0
    ratio_y = pps_y / max_pps if pps_y > 0 else 0.0

    remaining_x = steps_x
    remaining_y = steps_y
    acc_x = 0.0
    acc_y = 0.0

    while remaining_x > 0 or remaining_y > 0:
        if emergency_stop():
            break

        pulse_x = False
        pulse_y = False
        if remaining_x > 0:
            acc_x += ratio_x
        if remaining_y > 0:
            acc_y += ratio_y
        if remaining_x > 0 and acc_x >= 1.0:
            pulse_x = True
            acc_x -= 1.0
            remaining_x -= 1
        if remaining_y > 0 and acc_y >= 1.0:
            pulse_y = True
            acc_y -= 1.0
            remaining_y -= 1

        if pulse_x or pulse_y:
            step_pulses(pulse_x, pulse_y, half_period)
        else:
            time.sleep(period)

def move_x_to(target_x_cm: float, rpm_x: float = 60) -> None:

    global current_x_cm
    mapped_x = map_x(target_x_cm)
    delta_x = mapped_x - current_x_cm
    steps_x = round(abs(delta_x) * CM_TO_STEP_X)
    if steps_x == 0:
        return
    dir_x = 1 if delta_x >= 0 else 0
    move_xy_steps(steps_x, 0, rpm_x=rpm_x, rpm_y=0, dir_x=dir_x, dir_y=1)
    if not emergency_stop():
        current_x_cm = mapped_x


def move_y_to(target_y_cm: float, rpm_y: float = 60) -> None:
    global current_y_cm
    delta_y = target_y_cm - current_y_cm
    steps_y = round(abs(delta_y) * CM_TO_STEP_Y)
    if steps_y == 0:
        return
    dir_y = 1 if delta_y >= 0 else 0
    move_xy_steps(0, steps_y, rpm_x=0, rpm_y=rpm_y, dir_x=1, dir_y=dir_y)
    if not emergency_stop():
        current_y_cm = target_y_cm

def move_relative_xy(delta_x_cm: float, delta_y_cm: float,
                     rpm_x: float = 60, rpm_y: float = 60) -> None:
    global current_x_cm, current_y_cm

    steps_x = round(abs(delta_x_cm) * CM_TO_STEP_X)
    steps_y = round(abs(delta_y_cm) * CM_TO_STEP_Y)
    if steps_x == 0 and steps_y == 0:
        return

    dir_x = 1 if delta_x_cm >= 0 else 0
    dir_y = 1 if delta_y_cm >= 0 else 0
    move_xy_steps(steps_x, steps_y, rpm_x=rpm_x, rpm_y=rpm_y, dir_x=dir_x, dir_y=dir_y)
    time.sleep(0.05)
    if not emergency_stop():
        current_x_cm += delta_x_cm
        current_y_cm += delta_y_cm


def map_x(target_x_cm: float) -> float:
    if target_x_cm < 4.0:
        return target_x_cm
    return target_x_cm


def move_to_xy(target_x_cm: float, target_y_cm: float,
               rpm_x: float = 60, rpm_y: float = 60) -> None:
    global current_x_cm, current_y_cm
    mapped_x = map_x(target_x_cm)
    delta_x = mapped_x - current_x_cm
    delta_y = target_y_cm - current_y_cm
    move_relative_xy(delta_x, delta_y, rpm_x=rpm_x, rpm_y=rpm_y)

def servo_angle_from_absolute(angle: float) -> float:
    return max(0.0, min(180.0, angle + 100.0))


def duty_for_servo_angle(angle: float) -> float:
    servo_angle = servo_angle_from_absolute(angle)
    return 2.5 + (servo_angle / 180.0) * (12.5 - 2.5)


def set_servo_angle(angle: float, settle_s: float = 0.5):
    init_gpio()
    pwm = ensure_servo_pwm()
    pwm.ChangeDutyCycle(duty_for_servo_angle(angle))
    time.sleep(settle_s)
    pwm.ChangeDutyCycle(0)


def set_clean_servo_up(settle_s: float = 0.5) -> None:
    set_servo_angle(CLEAN_SERVO_UP_ANGLE, settle_s=settle_s)


def set_clean_servo_down(settle_s: float = 0.5) -> None:
    set_servo_angle(CLEAN_SERVO_DOWN_ANGLE, settle_s=settle_s)


# =====================================================================
# Detection-label -> cm conversion
# =====================================================================
@dataclass
class Spot:
    """One raw YOLO detection, expressed in board-cm.

    Extents (w_cm, h_cm) are kept so that disk packing can reason about
    the physical bounding box of each detection, not just its centroid.
    """
    image_index: int
    class_id: int
    xc_norm: float
    yc_norm: float
    w_norm: float
    h_norm: float
    board_x_cm: float
    board_y_cm: float
    w_cm: float
    h_cm: float
    stepper_x_cm: float
    stepper_y_cm: float

    @property
    def xmin_cm(self) -> float:
        return self.board_x_cm - self.w_cm / 2.0

    @property
    def xmax_cm(self) -> float:
        return self.board_x_cm + self.w_cm / 2.0

    @property
    def ymin_cm(self) -> float:
        return self.board_y_cm - self.h_cm / 2.0

    @property
    def ymax_cm(self) -> float:
        return self.board_y_cm + self.h_cm / 2.0


@dataclass
class CleaningTarget:
    """One sponge placement. May absorb one or many source spots."""
    board_x_cm: float            # center of merged footprint
    board_y_cm: float
    stepper_x_cm: float
    stepper_y_cm: float
    bbox_cm: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    source_spot_indices: list[int] = field(default_factory=list)

    @property
    def span_x_cm(self) -> float:
        return self.bbox_cm[2] - self.bbox_cm[0]

    @property
    def span_y_cm(self) -> float:
        return self.bbox_cm[3] - self.bbox_cm[1]


def load_label_file(label_path: Path, image_index: int) -> list[Spot]:
    """Read a YOLO .txt and convert each box into a Spot in board cm.

    YOLO line:  class_id  xc  yc  w  h  (all normalized).
    Conversion:
        board_x = origin_x + xc * IMAGE_WIDTH_CM
        board_y = origin_y + (1.0 - yc) * IMAGE_HEIGHT_CM
        w_cm    = w * IMAGE_WIDTH_CM
        h_cm    = h * IMAGE_HEIGHT_CM
    """
    spots: list[Spot] = []
    if not label_path.exists():
        print(f"Label file missing: {label_path}")
        return spots

    origin_x, origin_y = IMAGE_ORIGINS_CM[image_index]

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        xc, yc, w, h = map(float, parts[1:])

        board_x = origin_x + xc * IMAGE_WIDTH_CM
        board_y = origin_y + (1.0 - yc) * IMAGE_HEIGHT_CM
        w_cm = w * IMAGE_WIDTH_CM
        h_cm = h * IMAGE_HEIGHT_CM
        spots.append(
            Spot(
                image_index=image_index,
                class_id=class_id,
                xc_norm=xc,
                yc_norm=yc,
                w_norm=w,
                h_norm=h,
                board_x_cm=board_x,
                board_y_cm=board_y,
                w_cm=w_cm,
                h_cm=h_cm,
                stepper_x_cm=board_x + CAM_OFFSET_X_CM,
                stepper_y_cm=board_y + CAM_OFFSET_Y_CM,
            )
        )
    return spots


def sort_by_x(spots: list[Spot]) -> list[Spot]:
    """Visit detections strictly left-to-right by board X.
    Ties broken by Y so behavior is deterministic."""
    return sorted(spots, key=lambda s: (s.board_x_cm, s.board_y_cm))


# =====================================================================
# Disk-pack merging  --  raw spots -> cleaning targets
# =====================================================================
import math as _math


def _tile_oversize_spot(spot: Spot, sponge_cm: float) -> list[Spot]:
    """If a single YOLO box is larger than the sponge in either axis,
    split it into a grid of sub-spots so that every sub-spot fits under
    one sponge placement.  Returns the input unchanged if it already fits.

    The sub-spots share the source spot's image_index and class_id; the
    back-pointer to the original YOLO detection (via global index) is
    preserved upstream by disk_pack_spots.
    """
    over_x = spot.w_cm > sponge_cm
    over_y = spot.h_cm > sponge_cm
    if not (over_x or over_y):
        return [spot]

    n_x = max(1, _math.ceil(spot.w_cm / sponge_cm)) if over_x else 1
    n_y = max(1, _math.ceil(spot.h_cm / sponge_cm)) if over_y else 1
    sub_w = spot.w_cm / n_x
    sub_h = spot.h_cm / n_y

    subs: list[Spot] = []
    for ix in range(n_x):
        for iy in range(n_y):
            cx = spot.xmin_cm + sub_w * (ix + 0.5)
            cy = spot.ymin_cm + sub_h * (iy + 0.5)
            subs.append(
                Spot(
                    image_index=spot.image_index,
                    class_id=spot.class_id,
                    xc_norm=spot.xc_norm,   # preserved for provenance only
                    yc_norm=spot.yc_norm,
                    w_norm=spot.w_norm,
                    h_norm=spot.h_norm,
                    board_x_cm=cx,
                    board_y_cm=cy,
                    w_cm=sub_w,
                    h_cm=sub_h,
                    stepper_x_cm=cx + CAM_OFFSET_X_CM,
                    stepper_y_cm=cy + CAM_OFFSET_Y_CM,
                )
            )
    return subs


def disk_pack_spots(spots: list[Spot],
                    sponge_cm: float = SPONGE_EFFECTIVE_CM) -> list[CleaningTarget]:
    """Collapse raw YOLO detections into 2D disk-packed cleaning targets.

    Pre-step (oversize handling):
      Any single spot larger than `sponge_cm` in either axis is first
      tiled into a grid of sub-spots (each <= sponge_cm).  This
      guarantees that even a single huge YOLO detection is fully covered
      by multiple sponge placements rather than leaving strips uncleaned.

    Main algorithm (greedy 2D packing, left-to-right seed order):
      1. Sort remaining spots by (xmin, ymin).
      2. Seed a new target with the leftmost remaining spot.
      3. Repeatedly try to absorb another remaining spot into the target:
         tentatively extend the target's bounding box to cover it; accept
         iff the extended bbox still fits within `sponge_cm` in BOTH X and Y.
         Each successful absorption restarts the scan (bbox just grew).
      4. When no more spots can be absorbed, finalise the target.
      5. Repeat until no spots remain.

    Guarantees:
      * Every raw spot (including oversize ones, via tiling) is fully
        covered by at least one target.
      * Each target's span is <= sponge_cm in both axes, so a single
        centered sponge placement reaches every pixel of the target.
      * Large residues that do not fit under one sponge produce multiple
        adjacent targets whose combined coverage spans the entire residue.

    Provenance:
      Each target's `source_spot_indices` lists indices into the ORIGINAL
      `spots` list passed in (not the tiled intermediate list), so callers
      can map a target back to the YOLO detection(s) it serves.
    """
    if not spots:
        return []

    # Pre-step: tile oversize spots, remembering which ORIGINAL spot each
    # tile came from so source_spot_indices stays meaningful upstream.
    working: list[tuple[int, Spot]] = []   # (original_index, spot_or_subspot)
    for orig_idx, spot in enumerate(spots):
        for sub in _tile_oversize_spot(spot, sponge_cm):
            working.append((orig_idx, sub))

    targets: list[CleaningTarget] = []

    while working:
        working.sort(key=lambda item: (item[1].xmin_cm, item[1].ymin_cm))

        seed_orig_idx, seed = working.pop(0)
        group_indices: list[int] = [seed_orig_idx]
        xmin, ymin = seed.xmin_cm, seed.ymin_cm
        xmax, ymax = seed.xmax_cm, seed.ymax_cm

        # Greedy absorption with restart.
        absorbed_any = True
        while absorbed_any:
            absorbed_any = False
            for i, (cand_orig_idx, cand) in enumerate(working):
                new_xmin = min(xmin, cand.xmin_cm)
                new_ymin = min(ymin, cand.ymin_cm)
                new_xmax = max(xmax, cand.xmax_cm)
                new_ymax = max(ymax, cand.ymax_cm)
                if ((new_xmax - new_xmin) <= sponge_cm and
                        (new_ymax - new_ymin) <= sponge_cm):
                    xmin, ymin = new_xmin, new_ymin
                    xmax, ymax = new_xmax, new_ymax
                    group_indices.append(cand_orig_idx)
                    working.pop(i)
                    absorbed_any = True
                    break  # restart scan against the enlarged bbox

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        targets.append(
            CleaningTarget(
                board_x_cm=cx,
                board_y_cm=cy,
                stepper_x_cm=cx + CAM_OFFSET_X_CM,
                stepper_y_cm=cy + CAM_OFFSET_Y_CM,
                bbox_cm=(xmin, ymin, xmax, ymax),
                # Dedup in case an oversize spot's tiles fell into this target
                source_spot_indices=sorted(set(group_indices)),
            )
        )

    return targets


def sort_targets_by_x(targets: list[CleaningTarget]) -> list[CleaningTarget]:
    """Visit merged targets left-to-right; ties broken by Y."""
    return sorted(targets, key=lambda t: (t.board_x_cm, t.board_y_cm))


def group_targets_by_column(
    targets: list[CleaningTarget],
    tolerance_cm: float = X_COLUMN_TOLERANCE_CM,
) -> list[list[CleaningTarget]]:
    """Group cleaning targets into X-columns.

    Two targets share a column iff their stepper X coordinates differ by
    at most `tolerance_cm`. Columns are returned left-to-right. Within
    each column, targets are sorted by ascending stepper Y so smaller Y
    positions are cleaned before larger Y positions at the same X.

    Grouping uses a simple left-to-right sweep rather than clustering:
    when the next target's X is within `tolerance_cm` of the CURRENT
    COLUMN'S seed X, it joins that column. This keeps column X positions
    well-defined and avoids chain-drift where successive within-tolerance
    hops create an unbounded column.
    """
    if not targets:
        return []

    by_x = sorted(targets, key=lambda t: (t.stepper_x_cm, t.stepper_y_cm))
    columns: list[list[CleaningTarget]] = []
    column_seed_x: list[float] = []

    for t in by_x:
        if columns and abs(t.stepper_x_cm - column_seed_x[-1]) <= tolerance_cm:
            columns[-1].append(t)
        else:
            columns.append([t])
            column_seed_x.append(t.stepper_x_cm)

    # Within each column: smaller Y first, then larger Y.
    for col in columns:
        col.sort(key=lambda t: t.stepper_y_cm)

    return columns


def flatten_columns(columns: list[list[CleaningTarget]]) -> list[CleaningTarget]:
    """Flatten a list of columns into a single visit-order list."""
    out: list[CleaningTarget] = []
    for col in columns:
        out.extend(col)
    return out


def print_table(spots: list[Spot], header: str) -> None:
    print(f"\n{header}  ({len(spots)} spots)")
    print(f"{'#':>3} {'img':>3} {'xc':>6} {'yc':>6} "
          f"{'board(x,y)cm':>16} {'stepper(x,y)cm':>18}")
    for i, s in enumerate(spots):
        print(f"{i:>3} {s.image_index:>3} {s.xc_norm:>6.3f} {s.yc_norm:>6.3f} "
              f"({s.board_x_cm:6.2f},{s.board_y_cm:6.2f})   "
              f"({s.stepper_x_cm:7.2f},{s.stepper_y_cm:7.2f})")


def print_target_table(targets: list[CleaningTarget], header: str) -> None:
    print(f"\n{header}  ({len(targets)} targets)")
    print(f"{'#':>3} {'center(x,y)cm':>16} {'span(x,y)cm':>14} "
          f"{'stepper(x,y)cm':>18} {'sources':>8}")
    for i, t in enumerate(targets):
        print(f"{i:>3} ({t.board_x_cm:6.2f},{t.board_y_cm:6.2f}) "
              f"  ({t.span_x_cm:5.2f},{t.span_y_cm:5.2f})  "
              f"({t.stepper_x_cm:7.2f},{t.stepper_y_cm:7.2f})   "
              f"{len(t.source_spot_indices):>3} -> {t.source_spot_indices}")


# =====================================================================
# Merge-overlay visualisation
# =====================================================================
def _spot_indices_in_tile(target: CleaningTarget,
                          spots: list[Spot],
                          image_index: int) -> list[int]:
    """Return the global indices of source spots of `target` that came
    from the tile with the given image_index."""
    return [i for i in target.source_spot_indices
            if 0 <= i < len(spots) and spots[i].image_index == image_index]


def draw_merge_overlay(base_image_path: Path,
                       spots: list[Spot],
                       targets: list[CleaningTarget],
                       image_index: int,
                       out_path: Path,
                       sponge_diameter_cm: float = SPONGE_DIAMETER_CM) -> Path:
    """Draw merged sponge footprints on top of a YOLO overlay image.

    The base image should already contain YOLO's raw bounding boxes (e.g.
    the overlay PNG written by run_pair_preprocess_detect_4.py). This
    function adds green ellipses (physical sponge footprints) at each
    merged target center and labels them.

    Axes are computed independently per direction, so the footprint is
    rendered with correct physical proportions regardless of whether the
    image aspect ratio matches the tile's physical aspect ratio.
    """
    import cv2
    import numpy as np

    base_image_path = Path(base_image_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imdecode(np.fromfile(str(base_image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load overlay base image: {base_image_path}")

    h_px, w_px = img.shape[:2]
    origin_x, origin_y = IMAGE_ORIGINS_CM[image_index]
    px_per_cm_x = w_px / IMAGE_WIDTH_CM
    px_per_cm_y = h_px / IMAGE_HEIGHT_CM

    radius_px_x = int(round((sponge_diameter_cm / 2.0) * px_per_cm_x))
    radius_px_y = int(round((sponge_diameter_cm / 2.0) * px_per_cm_y))

    green = (0, 255, 0)       # BGR
    yellow = (0, 255, 255)
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Draw all targets that touch this tile.
    for t_idx, target in enumerate(targets, start=1):
        if not _spot_indices_in_tile(target, spots, image_index):
            continue

        cx_cm_local = target.board_x_cm - origin_x
        cy_cm_local = target.board_y_cm - origin_y
        cx_px = int(round(cx_cm_local * px_per_cm_x))
        cy_px = int(round((IMAGE_HEIGHT_CM - cy_cm_local) * px_per_cm_y))

        # Sponge footprint (physical 4 cm disk, rendered as ellipse iff
        # px/cm differs between axes).
        cv2.ellipse(
            img,
            center=(cx_px, cy_px),
            axes=(radius_px_x, radius_px_y),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=green,
            thickness=3,
        )

        # Center cross.
        cv2.drawMarker(
            img,
            position=(cx_px, cy_px),
            color=yellow,
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )

        # Label with target index and source-spot count.
        label = f"T{t_idx} (n={len(target.source_spot_indices)})"
        # Prefer to the right of the ellipse; fall back to the left if
        # that would place the label off-screen.
        label_x = cx_px + radius_px_x + 6
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        if label_x + text_w > w_px - 4:
            label_x = max(4, cx_px - radius_px_x - text_w - 6)
        label_y = max(text_h + 4, cy_px - radius_px_y - 4)
        label_pos = (label_x, label_y)
        # black outline then white fill for readability on any background
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, black, 4, cv2.LINE_AA)
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, white, 2, cv2.LINE_AA)

    suffix = out_path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, img)
    if not ok:
        raise RuntimeError(f"Failed to encode merge overlay: {out_path}")
    encoded.tofile(str(out_path))
    return out_path


# =====================================================================
# Auto-clean main
# =====================================================================
def auto_clean(label_left: Path, label_right: Path | None, dry_run: bool) -> int:
    spots: list[Spot] = []
    spots += load_label_file(label_left, image_index=0)
    if label_right is not None:
        spots += load_label_file(label_right, image_index=1)

    print_table(spots, "All raw detections")

    raw_targets = disk_pack_spots(spots)
    ordered = sort_targets_by_x(raw_targets)
    print_target_table(ordered, "Merged cleaning targets (before column grouping)")

    columns = group_targets_by_column(ordered)
    print(f"\nColumn-chained visit order: {len(columns)} column(s) "
          f"from {len(ordered)} target(s) "
          f"(tolerance={X_COLUMN_TOLERANCE_CM:.2f} cm)")
    for c_idx, col in enumerate(columns, start=1):
        col_x = sum(t.stepper_x_cm for t in col) / len(col)
        y_list = ", ".join(f"{t.stepper_y_cm:.2f}" for t in col)
        print(f"  Col {c_idx} @ x={col_x:.2f} cm  -> Y visit order: [{y_list}] (smallest Y first)")

    if dry_run or not columns:
        return 0

    try:
        set_clean_servo_up()
        total = sum(len(c) for c in columns)
        visited = 0
        for c_idx, col in enumerate(columns, start=1):
            col_x = sum(t.stepper_x_cm for t in col) / len(col)
            print(f"\n=== Column {c_idx}/{len(columns)} @ x={col_x:.2f} cm "
                  f"({len(col)} target(s)) ===")
            print(f"[col {c_idx}] xto {col_x:.2f} cm")
            move_x_to(col_x, rpm_x=RPM_X)

            for t in col:
                visited += 1
                print(f"[{visited}/{total}] yto {t.stepper_y_cm:.2f} cm")
                move_y_to(t.stepper_y_cm, rpm_y=RPM_Y)

                print(f"     [DC Motor] Cleaning for 3 seconds...")
                GPIO = _import_gpio()
                try:
                    set_clean_servo_down()
                    GPIO.output(DC_MOTOR_PIN, GPIO.HIGH)
                    time.sleep(3.0)
                    GPIO.output(DC_MOTOR_PIN, GPIO.LOW)
                finally:
                    GPIO.output(DC_MOTOR_PIN, GPIO.LOW)
                    set_clean_servo_up()

            # Y returns to 0 only once per column, after all its targets
            # have been visited from smaller Y to larger Y.
            print(f"[col {c_idx}] Yto0 (end of column)")
            move_y_to(0.0, rpm_y=RPM_Y)

        print("xto0")
        move_x_to(0.0, rpm_x=RPM_X)


    finally:
        global _gpio_ready, servo_pwm
        try:
            if servo_pwm is not None:
                release_servo_pwm()
                print("[System] Servo PWM stopped.")
            disable_driver()
            _import_gpio().cleanup()
            _gpio_ready = False 
            print("[System] GPIO Cleaned and flags reset.")
        except Exception as e:
            print(f"Cleanup error: {e}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drive stepper to YOLO detections (with disk-pack merging).")
    p.add_argument("--label-left",  type=Path, required=True,
                   help="YOLO label .txt for the left rectified image (image 1).")
    p.add_argument("--label-right", type=Path, default=None,
                   help="YOLO label .txt for the right rectified image (image 2).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned moves without driving the motor.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    raise SystemExit(auto_clean(args.label_left, args.label_right, args.dry_run))
