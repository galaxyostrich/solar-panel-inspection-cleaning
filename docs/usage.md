# Usage Guide

All scripts live in `src/` and load each other by filename, so **run every command from inside `src/`**.

```bash
cd src
```

## 1. Installation

**Laptop / PC** (detection server):

```bash
pip install -r ../requirements.txt
```

**Raspberry Pi** (capture + motor control):

```bash
pip install -r ../requirements-pi.txt
```

The Pi also needs `rpicam-still` (from Raspberry Pi OS `rpicam-apps`) and a connected Camera Module.
Copy `src/fpv_main_orchestrator.py` and `src/shuo.py` to the same folder on the Pi.

## 2. Full system run (Pi + laptop)

Both machines must be on the same network. Start the laptop first; it waits for one capture batch.

**Laptop:**

```bash
python fpv_main_orchestrator.py laptop --model-path ../models/best.pt --shuo-script shuo.py
```

**Raspberry Pi** (replace `<LAPTOP_IP>` with the laptop's LAN address):

```bash
python3 fpv_main_orchestrator.py pi --laptop-host <LAPTOP_IP> --shuo-script shuo.py --width 3280 --height 2464
```

Useful Pi options:

| Option | Default | Purpose |
|---|---|---|
| `--dry-run` | off | Capture + detect + print targets, but do not run the cleaning motors |
| `--capture-x-cms` | `9.6 17.6 25.5 33.2` | Rail positions (cm) for the four captures |
| `--clean-dwell-s` | `1.0` | Seconds the sponge spins on each target |
| `--x-column-tolerance-cm` | `0.3` | Targets within this X distance are cleaned as one column |
| `--port` | `5001` | TCP port (must match on both sides) |

Useful laptop options: `--conf` (default `0.1`), `--iou` (default `0.5`), `--work-dir` (default `~/fpv_work`).

Each laptop run writes a `session_YYYYMMDD_HHMMSS/` folder with `input/`, `processed/`, `labels/`,
`overlays/`, and `merge_overlays/`.

## 3. Laptop-only test (no hardware)

Run preprocessing + detection on a folder of 2–4 captures named in left-to-right order
(e.g. `img_1.jpg` … `img_4.jpg`):

```bash
python run_pair_preprocess_detect.py <image_folder> --model ../models/best.pt --out-dir <output_folder>
```

## 4. Calibration

The perspective-correction corner points for the four camera positions are stored in
`POINTS_BY_POSITION` inside `fix_2_preprocess.py`. To re-measure them after moving the camera:

```bash
python manual_calibrate_four_tiles.py --image-dir <folder_with_img_1..img_4>
```

Click the four panel corners on each image; the tool writes `manual_four_tile_outputs/tile*_rect.json`.
Copy the resulting points into `POINTS_BY_POSITION`.

Physical constants (panel size, offsets, GPIO pins, steps-per-cm, servo angles) are in the
`CALIBRATION` section at the top of `shuo.py`.

## 5. Hardware bring-up tests (Raspberry Pi)

```bash
python3 test_y_full_travel.py               # Y: 0 -> full travel, wait for Enter, back to 0
python3 test_move_xy_15_10.py --x 15 --y 10 # go to a simulated residue, lower servo, Enter returns home
python3 x_position_test.py --x-cm 10        # move X for calibration and set a servo angle
```
