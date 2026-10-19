"""
motor_controller.py  -- runs on the Raspberry Pi
=================================================
FINAL VERSION.

Role: the ONLY component that drives the actuators. Subscribes to the control
topic, maps command strings to PWM, enforces the ultrasonic safety reflex, and
runs the network watchdog. Optionally logs safety telemetry to InfluxDB.

Commands understood: forward | left | right | stop
(The earlier orbit/spin_search prototype commands have been removed; they were
part of the abandoned 'go around the marker' approach and are not used.)

Environment:
  US_FAIL_SAFE=1   stop when the ultrasonic sensor stops responding (default: off = v1 behaviour,
                   a silent sensor counts as 'path clear'; a warning is printed either way)
  CRUISE_SPEED     forward pulse width in us (default 1560); ESC_NEUTRAL for the stop value
  THROTTLE_DUTY    <1 pulses the throttle to drive slower than the ESC's own minimum,
                   e.g. 0.5 = half the time at cruise, half at neutral (default 1 = off)
  THROTTLE_PERIOD_S  length of one pulse cycle in s (default 0.2)
  INFLUX_TOKEN     enables InfluxDB telemetry (non-blocking batch writes); unset = off
"""

import os
import threading
import time

import paho.mqtt.client as mqtt
import pigpio

# --- Configuration & Pins ---
MQTT_BROKER = "localhost"
TOPIC_CMD = "rados/robot/control"

ESC_PIN = 18
SERVO_PIN = 12
TRIG_PIN = 23
ECHO_PIN = 24

# --- PWM Values (calibrated baseline) ---
# The ESC learns its neutral from the first signal after power-on: start this script
# BEFORE switching the ESC on. Values are settable without editing the file, e.g.
#   CRUISE_SPEED=1620 python3 motor_controller.py
# On the floor the car needs more throttle than with the wheels in the air.
ESC_NEUTRAL = int(os.environ.get("ESC_NEUTRAL", "1500"))      # true stop on this ESC
CRUISE_SPEED = int(os.environ.get("CRUISE_SPEED", "1530"))    # slow forward: measured on the
# floor 17.09 - 1560 was too fast to control at ~1 decision/s, 1530 rolls slowly and steadily
# A brushed ESC has a minimum throttle below which the car does not move at all. To drive
# slower than that, pulse between cruise and neutral and let the car coast in between.
THROTTLE_DUTY = float(os.environ.get("THROTTLE_DUTY", "1"))
THROTTLE_PERIOD_S = float(os.environ.get("THROTTLE_PERIOD_S", "0.2"))
SERVO_CENTER = int(os.environ.get("SERVO_CENTER", "1500"))
SERVO_LEFT = int(os.environ.get("SERVO_LEFT", "1200"))    # full lock; raise for gentler turns
SERVO_RIGHT = int(os.environ.get("SERVO_RIGHT", "1800"))   # full lock; lower for gentler turns
# Proportional steering: "go:<v>" with v in [-1,1] maps to SERVO_CENTER + v*STEER_GAIN_US.
STEER_GAIN_US = int(os.environ.get("STEER_GAIN_US", "150"))

# --- Safety Thresholds ---
STOP_DISTANCE_CM = 30.0   # stop if obstacle closer than 30 cm
WATCHDOG_TIMEOUT = 0.6    # stop if no command for 0.6 s

# --- Ultrasonic sensor (HC-SR04) ---
# The echo line normally rises ~0.5 ms after the trigger. If nothing is in
# range (> ~4 m) the sensor still raises it and holds it for ~38 ms.
#   echo never rises within ECHO_TIMEOUT_S -> sensor not responding (fault)
#   echo stays high longer than that       -> nothing in range: path clear
ECHO_TIMEOUT_S = 0.03
OUT_OF_RANGE = float("inf")
SENSOR_FAULT_N = 20                                        # consecutive faults (~2 s) -> sensor fault
FAIL_SAFE = os.environ.get("US_FAIL_SAFE", "0") == "1"     # on sensor fault: treat path as blocked

# --- InfluxDB (optional, never in source: export INFLUX_TOKEN=...) ---
INFLUX_URL = "https://eu-central-1-1.aws.cloud2.influxdata.com"
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "")
INFLUX_ORG = "vesrad"
INFLUX_BUCKET = "rasp-pi"

# --- Globals ---
driving = False           # a drive command is active (used by the throttle pulser)
path_clear = True
sensor_ok = True
last_msg_time = time.time()
current_dist = OUT_OF_RANGE

# --- Hardware & Influx Setup ---
pi = pigpio.pi()
pi.set_mode(TRIG_PIN, pigpio.OUTPUT)
pi.set_mode(ECHO_PIN, pigpio.INPUT)

write_api = None
if INFLUX_TOKEN:
    # Batching (non-blocking) writes: a slow or unreachable InfluxDB must never
    # delay the safety loop. v1 wrote synchronously to the cloud every 100 ms.
    from influxdb_client import InfluxDBClient, Point, WriteOptions
    write_api = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG).write_api(
        write_options=WriteOptions(batch_size=50, flush_interval=1000))


def stop_robot():
    pi.set_servo_pulsewidth(ESC_PIN, ESC_NEUTRAL)
    pi.set_servo_pulsewidth(SERVO_PIN, SERVO_CENTER)


def measure_distance():
    """Distance in cm, OUT_OF_RANGE if nothing is in range, None if the sensor did not respond."""
    pi.gpio_trigger(TRIG_PIN, 10)
    start = time.time()
    while pi.read(ECHO_PIN) == 0:
        if time.time() - start > ECHO_TIMEOUT_S:
            return None                          # echo never started: sensor fault / unplugged
    pulse_start = time.time()
    while pi.read(ECHO_PIN) == 1:
        if time.time() - pulse_start > ECHO_TIMEOUT_S:
            return OUT_OF_RANGE                  # no reflection within range
    pulse_end = time.time()
    return round(((pulse_end - pulse_start) * 34300) / 2, 2)


def safety_loop():
    """Background thread: monitor the ultrasonic sensor and enforce safety."""
    global path_clear, current_dist, sensor_ok
    faults = 0
    while True:
        dist = measure_distance()
        if dist is None:
            faults += 1
            if faults == SENSOR_FAULT_N:
                sensor_ok = False
                print("!!! ULTRASONIC SENSOR NOT RESPONDING%s" % (
                    " - stopping" if FAIL_SAFE else " - treating path as clear (US_FAIL_SAFE=1 stops instead)"))
                if FAIL_SAFE:
                    stop_robot()
        else:
            if not sensor_ok:
                print("Ultrasonic sensor responding again.")
            faults, sensor_ok, current_dist = 0, True, dist

        blocked = current_dist < STOP_DISTANCE_CM or (FAIL_SAFE and not sensor_ok)
        if blocked:
            if path_clear and sensor_ok:
                print(f"!!! OBSTACLE DETECTED: {current_dist} cm. Emergency Stop.")
                stop_robot()
            path_clear = False
        else:
            path_clear = True

        if write_api is not None:
            try:
                p = Point("robot_safety").field("dist", min(current_dist, 400.0)) \
                                         .field("clear", int(path_clear)) \
                                         .field("sensor_ok", int(sensor_ok))
                write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
            except Exception:
                pass

        time.sleep(0.1)


# --- MQTT command handling ---
last_blocked_cmd = None
last_block_print_time = 0


def on_message(client, userdata, msg):
    global last_msg_time, last_blocked_cmd, last_block_print_time, driving
    command = msg.payload.decode().strip()
    last_msg_time = time.time()

    # Safety veto: block forward motion when the path is not clear
    if not path_clear and (command in ["forward", "left", "right"] or command.startswith("go:")):
        if command != last_blocked_cmd or time.time() - last_block_print_time > 1.0:
            print(f"Ignoring '{command}' - Path Blocked.")
            last_blocked_cmd = command
            last_block_print_time = time.time()
        driving = False
        stop_robot()
        return

    if command.startswith("go:"):                      # proportional steering
        try:
            v = max(-1.0, min(1.0, float(command[3:])))
        except ValueError:
            return
        pi.set_servo_pulsewidth(SERVO_PIN, SERVO_CENTER + v * STEER_GAIN_US)
        driving = True
        if THROTTLE_DUTY >= 1.0:
            pi.set_servo_pulsewidth(ESC_PIN, CRUISE_SPEED)
        return

    steering = {"forward": SERVO_CENTER, "left": SERVO_LEFT, "right": SERVO_RIGHT}
    if command in steering:
        pi.set_servo_pulsewidth(SERVO_PIN, steering[command])
        driving = True
        if THROTTLE_DUTY >= 1.0:
            pi.set_servo_pulsewidth(ESC_PIN, CRUISE_SPEED)   # throttle_loop pulses it otherwise
    elif command == "stop":
        driving = False
        stop_robot()


def throttle_loop():
    """Pulse the throttle when THROTTLE_DUTY < 1, so the car can drive slower than the
    ESC's minimum. Steering stays with on_message; only the ESC is pulsed here."""
    on_s = THROTTLE_PERIOD_S * THROTTLE_DUTY
    off_s = THROTTLE_PERIOD_S - on_s
    while True:
        if driving and path_clear:
            pi.set_servo_pulsewidth(ESC_PIN, CRUISE_SPEED)
            time.sleep(on_s)
            if driving and path_clear:
                pi.set_servo_pulsewidth(ESC_PIN, ESC_NEUTRAL)
                time.sleep(off_s)
        else:
            time.sleep(0.02)


def watchdog_loop():
    """Stop the vehicle if no command has arrived within the timeout."""
    while True:
        if time.time() - last_msg_time > WATCHDOG_TIMEOUT:
            globals()["driving"] = False
            stop_robot()
        time.sleep(0.1)


# --- Start Up ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = lambda c, u, f, rc, p: c.subscribe(TOPIC_CMD)
client.on_message = on_message

if __name__ == "__main__":
    stop_robot()
    threading.Thread(target=safety_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    if THROTTLE_DUTY < 1.0:
        print("Throttle pulsing: %.0f%% duty, %.2f s cycle" % (THROTTLE_DUTY * 100, THROTTLE_PERIOD_S))
        threading.Thread(target=throttle_loop, daemon=True).start()
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        stop_robot()
        pi.stop()
