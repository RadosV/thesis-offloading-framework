#!/usr/bin/env python3
"""
Manual drive test (Pi). Drives the servo and ESC directly - no MQTT, no perception,
no motor_controller.py. Use it to check wiring, linkage and ESC arming by hand.

    python3 manual_drive.py

    a / d   steer left / right (steps of 50 us)
    s       centre the steering
    w / x   throttle forward / reverse (steps of 20 us)
    space   throttle to neutral
    r       re-arm helper: hold neutral so you can power-cycle the ESC
    p       print the current pulse widths
    q       quit (leaves both pins at neutral, NOT at 0)

Stop motor_controller.py first, or the two will fight over the same pins.
The ESC must be switched ON: the steering servo takes its power from the ESC's BEC.
"""
import sys
import termios
import time
import tty

import pigpio

ESC_PIN, SERVO_PIN = 18, 12
NEUTRAL = 1500
SERVO_MIN, SERVO_MAX = 1200, 1800     # full lock either way
ESC_MIN, ESC_MAX = 1400, 1600         # deliberately gentle: +-100 us around neutral
SERVO_STEP, ESC_STEP = 50, 20


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("pigpiod is not running - start it with: sudo pigpiod")
    servo, esc = NEUTRAL, NEUTRAL
    pi.set_servo_pulsewidth(SERVO_PIN, servo)
    pi.set_servo_pulsewidth(ESC_PIN, esc)
    print(__doc__)
    print("servo %d us | esc %d us   (both at neutral)" % (servo, esc))
    try:
        while True:
            k = getch().lower()
            if k == "q":
                break
            elif k == "a":
                servo = max(SERVO_MIN, servo - SERVO_STEP)
            elif k == "d":
                servo = min(SERVO_MAX, servo + SERVO_STEP)
            elif k == "s":
                servo = NEUTRAL
            elif k == "w":
                esc = min(ESC_MAX, esc + ESC_STEP)
            elif k == "x":
                esc = max(ESC_MIN, esc - ESC_STEP)
            elif k == " ":
                esc = NEUTRAL
            elif k == "r":
                servo = esc = NEUTRAL
                pi.set_servo_pulsewidth(SERVO_PIN, servo)
                pi.set_servo_pulsewidth(ESC_PIN, esc)
                print("\r holding neutral - switch the ESC OFF, then ON, wait for the beeps ")
                continue
            elif k == "p":
                print("\r servo %d us | esc %d us                    " % (servo, esc))
                continue
            else:
                continue
            pi.set_servo_pulsewidth(SERVO_PIN, servo)
            pi.set_servo_pulsewidth(ESC_PIN, esc)
            print("\r servo %4d us %-6s | esc %4d us %-8s " % (
                servo, "LEFT" if servo < NEUTRAL else "RIGHT" if servo > NEUTRAL else "centre",
                esc, "fwd" if esc > NEUTRAL else "rev" if esc < NEUTRAL else "neutral"), end="")
            sys.stdout.flush()
    finally:
        # leave a valid neutral signal so the ESC keeps its arming
        pi.set_servo_pulsewidth(SERVO_PIN, NEUTRAL)
        pi.set_servo_pulsewidth(ESC_PIN, NEUTRAL)
        time.sleep(0.2)
        pi.stop()
        print("\nboth pins left at neutral (1500 us)")


if __name__ == "__main__":
    main()
