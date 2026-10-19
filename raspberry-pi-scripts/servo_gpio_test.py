#!/usr/bin/env python3
"""
Servo / GPIO bench test (Pi). Isolates "steering stuck at one end" into
signal problem vs. servo vs. mechanics. Touches ONLY the steering pin - the
ESC is never driven, so the car cannot drive off.

    python3 servo_gpio_test.py              # steering servo on GPIO 12
    python3 servo_gpio_test.py --pin 13     # same servo moved to another pin
    python3 servo_gpio_test.py --step 1400 1600   # narrow sweep around centre

Stop the motor controller first, otherwise its watchdog keeps re-centring.
Watch the SERVO HORN, not the wheels:
  horn moves, wheels do not  -> linkage / horn screw (mechanical)
  horn does not move at all  -> no signal, no power, or dead servo
  horn moves but jitters/hums-> power sag (battery / BEC too weak)
"""
import argparse
import time

import pigpio

CENTER, LEFT, RIGHT = 1500, 1200, 1800


def announce(text):
    print("\n=== %s" % text, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", type=int, default=12, help="steering servo GPIO (default 12)")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds per position (default 3)")
    ap.add_argument("--step", nargs=2, type=int, metavar=("FROM", "TO"), default=(1200, 1800))
    args = ap.parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        raise SystemExit("pigpiod is not running: sudo pigpiod")
    print("pigpio connected, hardware revision %s, steering pin GPIO %d" % (pi.get_hardware_revision(), args.pin))

    try:
        announce("1. centre (1500 us) - the wheels should point straight ahead")
        pi.set_servo_pulsewidth(args.pin, CENTER)
        time.sleep(args.hold)

        announce("2. left (1200) -> centre -> right (1800) -> centre")
        for value in (LEFT, CENTER, RIGHT, CENTER):
            print("   %d us" % value, flush=True)
            pi.set_servo_pulsewidth(args.pin, value)
            time.sleep(args.hold)

        announce("3. slow sweep %d -> %d us in 50 us steps (find the real centre)" % tuple(args.step))
        low, high = sorted(args.step)
        for value in range(low, high + 1, 50):
            print("   %d us" % value, flush=True)
            pi.set_servo_pulsewidth(args.pin, value)
            time.sleep(0.6)

        announce("4. pulses OFF - a healthy servo goes slack and stops pushing")
        pi.set_servo_pulsewidth(args.pin, 0)
        time.sleep(args.hold)
        print("   still pushing hard with no signal? -> the servo itself is faulty")

        announce("5. back to centre")
        pi.set_servo_pulsewidth(args.pin, CENTER)
        time.sleep(args.hold)
    finally:
        pi.set_servo_pulsewidth(args.pin, 0)
        pi.stop()
        print("\ndone, pulses off")


if __name__ == "__main__":
    main()
