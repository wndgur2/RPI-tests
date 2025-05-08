CHANNEL_SERVO_XZ = 0
CHANNEL_SERVO_YZ = 1
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

LASER_X = 0
LASER_Y = 0
LASER_Z = 0

PIN_LASER = 17

PID_P = 1

import math
import time
from adafruit_servokit import ServoKit
from gpiozero import LED

class Turret:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.servo_xz = self.kit.servo[CHANNEL_SERVO_XZ]
        self.servo_yz = self.kit.servo[CHANNEL_SERVO_YZ]
        self.servo_xz.set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        self.servo_yz.set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        self.servos = [self.servo_xz, self.servo_yz]
        self.laser = LED(PIN_LASER)
        self.servo_xz.angle = 0
        self.servo_yz.angle = 180

    def look_at(self, x, y, z):
        angle_xz = self.calculate_angle(x, z)
        angle_yz = self.calculate_angle(y, z)
                
        self.servo_xz.angle = pid(angle_xz)
        self.servo_yz.angle = pid(180-angle_yz)

    def calculate_angle(self, x, y):
        if x == 0 and y == 0:
            return 0
        angle = math.degrees(math.atan2(y, x))
        return angle

    def off(self):
        self.kit.servo[CHANNEL_SERVO_XZ].angle = 0
        self.kit.servo[CHANNEL_SERVO_YZ].angle = 0
        self.laser.off()

def pid(angle):
    angle /= PID_P

    if angle > 180:
        angle = 180
    elif angle < 0:
        angle = 0
    return angle

turret = Turret()

# ------Example usage------

target_x = -80
target_y = 0
target_z = 100

turret.look_at(target_x, target_y, target_z)

time.sleep(2)
# -------------------------

turret.off()