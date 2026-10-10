from __future__ import annotations

import argparse

import shuo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move to a simulated salt residue, lower servo, then return home after Enter."
    )
    parser.add_argument("--x", type=float, default=15.0, help="Target X coordinate in cm.")
    parser.add_argument("--y", type=float, default=10.6, help="Target Y coordinate in cm.")
    parser.add_argument("--rpm-x", type=float, default=20.0, help="X axis motor speed.")
    parser.add_argument("--rpm-y", type=float, default=20.0, help="Y axis motor speed.")
    parser.add_argument("--no-servo", action="store_true", help="Do not initialize or move the cleaning servo.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print(f"Configured panel size: X={shuo.PANEL_WIDTH_CM:.2f} cm, Y={shuo.IMAGE_HEIGHT_CM:.2f} cm")
    print(f"Moving to simulated salt coordinate: X={args.x:.2f} cm, Y={args.y:.2f} cm")
    print("Press Ctrl+C for emergency stop from the terminal if needed.")

    shuo.init_gpio()
    try:
        if not args.no_servo and hasattr(shuo, "set_clean_servo_up"):
            shuo.set_clean_servo_up()

        shuo.move_x_to(args.x, rpm_x=args.rpm_x)
        shuo.move_y_to(args.y, rpm_y=args.rpm_y)

        if not args.no_servo and hasattr(shuo, "set_clean_servo_down"):
            print("Lowering servo")
            shuo.set_clean_servo_down()

        input("Arrived with servo down. Press Enter to raise servo and return X/Y to 0...")

        if not args.no_servo and hasattr(shuo, "set_clean_servo_up"):
            print("Raising servo")
            shuo.set_clean_servo_up()

        print("Returning Y to 0 cm")
        shuo.move_y_to(0.0, rpm_y=args.rpm_y)
        print("Returning X to 0 cm")
        shuo.move_x_to(0.0, rpm_x=args.rpm_x)
        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Returning Y/X to 0 if possible...")
        try:
            shuo.move_y_to(0.0, rpm_y=args.rpm_y)
            shuo.move_x_to(0.0, rpm_x=args.rpm_x)
        except Exception as exc:
            print(f"Return-home failed: {exc}")
        return 130
    finally:
        try:
            if not args.no_servo and hasattr(shuo, "set_clean_servo_up"):
                shuo.set_clean_servo_up()
            if hasattr(shuo, "release_servo_pwm"):
                shuo.release_servo_pwm()
            shuo.disable_driver()
            shuo._import_gpio().cleanup()
            shuo._gpio_ready = False
        except Exception as exc:
            print(f"Cleanup warning: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
