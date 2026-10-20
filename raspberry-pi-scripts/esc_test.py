"""
Standalone ESC/servo tester. No MQTT, no vision, no threads.
Run on the Pi:  python3 esc_test.py
Finds the neutral and forward pulse widths of the ESC and servo.
"""
import pigpio
import time

ESC_PIN = 18
SERVO_PIN = 12

pi = pigpio.pi()

def esc(val):
    pi.set_servo_pulsewidth(ESC_PIN, val)

def steer(val):
    pi.set_servo_pulsewidth(SERVO_PIN, val)

try:
    print("Centering steering at 1500")
    steer(1500)
    time.sleep(1)

    print("\n--- ESC ARMING ---")
    print("Setting NEUTRAL 1500 for 3s. Wheels should be COMPLETELY still.")
    esc(1500)
    time.sleep(3)

    for val in [1520, 1540, 1560, 1580, 1600]:
        print(f"\nESC = {val}  (watch the wheels for 2.5s)")
        esc(val)
        time.sleep(2.5)
        print("   back to 1500 (stop)")
        esc(1500)
        time.sleep(1.5)

    print("\nDone. Stopping.")
    esc(1500)
    steer(1500)
    time.sleep(1)

finally:
    esc(1500)
    steer(1500)
    pi.set_servo_pulsewidth(ESC_PIN, 0)
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()