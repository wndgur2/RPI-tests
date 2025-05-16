# 서보 컨트롤러 채널 번호
CHANNEL_SERVO_XY = 0
CHANNEL_SERVO_YZ = 1

# 서보 모터 6221MG의 최소, 최대 펄스 폭
# 500us ~ 2500us
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# 레이저의 초기 위치 (카메라와의 거리)
LASER_X = 0
LASER_Y = 0
LASER_Z = 0

# 레이저 GPIO 핀 번호
PIN_LASER = 17

# PID 제어를 위한 비례 상수 (1이면 영향 없음)
PID_P = 1

import math
import time
from adafruit_servokit import ServoKit
from gpiozero import LED

class Turret:
    def __init__(self):
        # 서보 컨트롤러 객체 생성
        self.kit = ServoKit(channels=16)

        # xz평면, yz평면 서보 모터 초기화
        self.servo_xy = self.kit.servo[CHANNEL_SERVO_XY]
        self.servo_yz = self.kit.servo[CHANNEL_SERVO_YZ]
        self.servo_xy.set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        self.servo_yz.set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        self.servos = [self.servo_xy, self.servo_yz]
        self.servo_yz.angle = 0
        self.servo_xy.angle = 0

        # 레이저 초기화
        self.laser = LED(PIN_LASER)

    def look_at(self, x, y, z):
        angle_xy = self.calculate_angle(x, y)
        angle_yz = self.calculate_angle(y, z)
        
        self.servo_xy.angle = pid(angle_xy)
        self.servo_yz.angle = pid(angle_yz)

    def calculate_angle(self, x, y):
        if x == 0 and y == 0:
            return 0
        angle = math.degrees(math.atan2(y, x))
        return angle

    def off(self):
        self.kit.servo[CHANNEL_SERVO_XY].angle = 0
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
# 시간 측정
print(turret.kit)
turret.look_at(1200, 50, 100)

time.sleep(1)

# -------------------------

turret.off()