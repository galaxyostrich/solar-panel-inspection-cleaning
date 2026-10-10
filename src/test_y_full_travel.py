from __future__ import annotations

import argparse

import shuo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move Y from 0 to the configured max Y, wait for Enter, then return Y to 0."
    )
    parser.add_argument("--max-y", type=float, default=shuo.IMAGE_HEIGHT_CM, help="Y max coordinate in cm.")
    parser.add_argument("--rpm-y", type=float, default=20.0, help="Y axis motor speed.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print(f"Configured panel Y length: {shuo.IMAGE_HEIGHT_CM:.2f} cm")
    print(f"Moving Y from 0.00 cm to {args.max_y:.2f} cm")
    print("Press Ctrl+C for emergency stop from the terminal if needed.")

    shuo.init_gpio()
    try:
        shuo.move_y_to(args.max_y, rpm_y=args.rpm_y)
        input("Y reached max. Press Enter to return Y to 0...")
        print("Returning Y to 0 cm")
        shuo.move_y_to(0.0, rpm_y=args.rpm_y)
        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Returning Y to 0 if possible...")
        try:
            shuo.move_y_to(0.0, rpm_y=args.rpm_y)
        except Exception as exc:
            print(f"Return-home failed: {exc}")
        return 130
    finally:
        try:
            shuo.disable_driver()
            shuo._import_gpio().cleanup()
            shuo._gpio_ready = False
        except Exception as exc:
            print(f"Cleanup warning: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
