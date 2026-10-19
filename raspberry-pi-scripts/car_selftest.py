#!/usr/bin/env python3
"""
Car self-test (Pi): checks every link in the chain and says which one is broken.

    python3 car_selftest.py            # all checks, asks before moving anything
    python3 car_selftest.py --no-move  # only the checks that never drive the car

Order of checks: pigpiod - MQTT broker - ultrasonic - camera - edge brain -
steering servo - ESC/drive - end-to-end command path (via the motor controller).

Stop motor_controller.py before running this: it re-centres the steering and
would fight the direct servo/ESC tests. The end-to-end check at the end asks you
to start it again.
"""
import argparse
import subprocess
import sys
import time

ESC_PIN, SERVO_PIN, TRIG_PIN, ECHO_PIN = 18, 12, 23, 24
ESC_NEUTRAL, CRUISE = 1500, 1530
TOPIC_CMD, TOPIC_CLOUD = "rados/robot/control", "rados/robot/cloud_cmd"
results = []


def report(name, ok, detail=""):
    print("   %-4s %s %s" % ("OK" if ok else "FAIL", name, detail))
    results.append((name, ok))
    return ok


def head(text):
    print("\n=== %s" % text)


def ask(text):
    try:
        input("   %s (ENTER) " % text)
    except EOFError:
        pass


def check_pigpio():
    head("1. pigpio daemon")
    import pigpio
    pi = pigpio.pi()
    if not pi.connected:
        report("pigpiod", False, "-> start it with: sudo pigpiod")
        return None
    report("pigpiod", True, "hardware revision %s" % pi.get_hardware_revision())
    return pi


def check_broker():
    head("2. MQTT broker (mosquitto on the Pi)")
    import paho.mqtt.client as mqtt
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    try:
        client.connect("localhost", 1883, 5)
        client.loop_start()
        report("broker", True, "connected to localhost:1883")
        return client
    except Exception as e:
        report("broker", False, "%s -> sudo systemctl start mosquitto" % e)
        return None


def check_ultrasonic(pi):
    head("3. ultrasonic sensor (blocks ALL driving when something is within 30 cm)")
    import pigpio
    values, faults = [], 0
    for _ in range(10):
        pi.gpio_trigger(TRIG_PIN, 10)
        start = time.time()
        while pi.read(ECHO_PIN) == 0:
            if time.time() - start > 0.03:
                faults += 1
                break
        else:
            t0 = time.time()
            while pi.read(ECHO_PIN) == 1:
                if time.time() - t0 > 0.03:
                    break
            values.append(round((time.time() - t0) * 34300 / 2, 1))
        time.sleep(0.06)
    if faults == 10:
        return report("sensor", False, "no echo at all -> check wiring / voltage divider")
    near = [v for v in values if v < 30]
    report("sensor", True, "readings %s cm (%d no-echo)" % (values[:6], faults))
    if near:
        report("path clear", False, "something is within 30 cm -> the car will refuse to drive")
    else:
        report("path clear", True)


def check_camera():
    head("4. camera")
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return report("camera", False, "cannot open /dev/video0 -> is it plugged in?")
    ok, frame = cap.read()
    cap.release()
    report("camera", bool(ok), "frame %s" % (str(frame.shape) if ok else "none",))


def check_edge_brain(client):
    head("5. edge brain on the PC (listening for its decisions for 5 s)")
    seen = []
    client.subscribe(TOPIC_CLOUD)
    client.on_message = lambda c, u, m: seen.append(m.payload[:40])
    time.sleep(5)
    client.unsubscribe(TOPIC_CLOUD)
    client.on_message = None
    if seen:
        report("edge brain", True, "%d decisions, e.g. %s" % (len(seen), seen[-1]))
    else:
        report("edge brain", False, "silent -> is pc_edge_brain.py running on the PC, "
                                    "and is the manager streaming frames?")


def check_servo(pi):
    head("6. steering servo (watch the SERVO HORN, not the wheels)")
    ask("ready?")
    for value, text in ((1500, "centre"), (1200, "left"), (1800, "right"), (1500, "centre")):
        print("   %s (%d us)" % (text, value), flush=True)
        pi.set_servo_pulsewidth(SERVO_PIN, value)
        time.sleep(1.5)
    answer = input("   did the horn move left and right? [y/N] ").strip().lower()
    report("servo", answer == "y", "" if answer == "y" else
           "-> horn still: wiring/power or dead servo; horn moves but wheels do not: linkage")


def check_esc(pi):
    head("7. ESC / drive motor (car on the floor with space, or rear wheels lifted)")
    ask("ready?")
    pi.set_servo_pulsewidth(ESC_PIN, ESC_NEUTRAL)
    time.sleep(2)
    print("   neutral %d us - wheels must be still" % ESC_NEUTRAL, flush=True)
    still = input("   were the wheels still at neutral? [y/N] ").strip().lower()
    report("esc neutral", still == "y", "" if still == "y" else
           "-> the ESC armed on a wrong signal: switch it OFF, hold 1500 us, switch ON, wait for beeps")
    print("   cruise %d us for 2 s" % CRUISE, flush=True)
    pi.set_servo_pulsewidth(ESC_PIN, CRUISE)
    time.sleep(2)
    pi.set_servo_pulsewidth(ESC_PIN, ESC_NEUTRAL)
    answer = input("   did it drive FORWARD? [y/N] ").strip().lower()
    report("esc drive", answer == "y", "" if answer == "y" else
           "-> backwards or nothing: same re-arming procedure; battery charged?")


def check_end_to_end(client):
    head("8. end-to-end: motor controller reacting to commands")
    running = subprocess.run(["pgrep", "-f", "motor_controller.py"],
                             stdout=subprocess.PIPE).stdout.strip()
    if not report("motor_controller.py running", bool(running),
                  "" if running else "-> start it in another window, then re-run this test"):
        return
    ask("sending 'forward' for 3 s - ready?")
    end = time.time() + 3
    while time.time() < end:
        client.publish(TOPIC_CMD, "forward")
        time.sleep(0.1)
    client.publish(TOPIC_CMD, "stop")
    answer = input("   did the car drive? [y/N] ").strip().lower()
    report("command path", answer == "y", "" if answer == "y" else
           "-> check the motor controller window for 'Path Blocked' (ultrasonic) or errors")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-move", action="store_true", help="skip the servo, ESC and command tests")
    args = ap.parse_args()

    pi = check_pigpio()
    client = check_broker()
    if pi:
        check_ultrasonic(pi)
    check_camera()
    if client:
        check_edge_brain(client)
    if pi and not args.no_move:
        check_servo(pi)
        check_esc(pi)
        if client:
            check_end_to_end(client)

    head("summary")
    for name, ok in results:
        print("   %-28s %s" % (name, "OK" if ok else "FAIL"))
    bad = [n for n, ok in results if not ok]
    print("\n%s" % ("all checks passed" if not bad else "broken: " + ", ".join(bad)))
    if pi:
        pi.set_servo_pulsewidth(ESC_PIN, 0)
        pi.set_servo_pulsewidth(SERVO_PIN, 0)
        pi.stop()
    if client:
        client.loop_stop()
        client.disconnect()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
