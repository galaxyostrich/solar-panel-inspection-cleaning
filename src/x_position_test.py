from __future__ import annotations

import argparse
import time
from pathlib import Path


def import_module_from_path(path: Path, module_name: str):
    import importlib.util
    import sys

    path = path.resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Move X axis to a target cm position for calibration.")
    parser.add_argument("--shuo-script", type=Path, default=Path("./shuo.py"))
    parser.add_argument("--x-cm", type=float, default=10.0)
    parser.add_argument("--rpm-x", type=float, default=20.0)
    parser.add_argument("--servo-angle", type=float, default=-90.0)
    parser.add_argument(
        "--hold-driver",
        action="store_true",
        help="Keep stepper driver enabled after moving. Default disables driver.",
    )
    parser.add_argument(
        "--return-home",
        action="store_true",
        help="Return to X=0 after pressing Enter.",
    )
    args = parser.parse_args()

    shuo = import_module_from_path(args.shuo_script, "shuo_x_position_test")

    try:
        shuo.init_gpio()
        print(f"Moving X to {args.x_cm:.2f} cm at rpm_x={args.rpm_x:.1f}", flush=True)
        shuo.move_x_to(args.x_cm, rpm_x=args.rpm_x)
        print(f"Reached commanded X={args.x_cm:.2f} cm.", flush=True)

        if hasattr(shuo, "set_servo_angle"):
            print(f"Moving servo to {args.servo_angle:.1f} degrees.", flush=True)
            shuo.set_servo_angle(args.servo_angle)
        else:
            print("Warning: shuo.py has no set_servo_angle().", flush=True)

        if args.hold_driver:
            print("Driver is still enabled.", flush=True)
        else:
            shuo.disable_driver()
            print("Driver disabled.", flush=True)

        if args.return_home:
            input("Press Enter to return X to 0.00 cm...")
            shuo.init_gpio()
            print("Returning X to 0.00 cm", flush=True)
            shuo.move_x_to(0.0, rpm_x=args.rpm_x)
    finally:
        try:
            if not args.hold_driver:
                shuo.disable_driver()
            shuo._import_gpio().cleanup()
            shuo._gpio_ready = False
        except Exception:
            pass
        time.sleep(0.1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
