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

OFFSET_Y = 151.5 # 빼고
OFFSET_Z = 30 # 빼고

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
        self.servo_xy.angle = 0
        self.servo_yz.angle = 0

        # 레이저 초기화
        self.laser = LED(PIN_LASER)
        
    def look_at(self, x, y, z):
        y = y - OFFSET_Y
        z = z - OFFSET_Z
        print('[TURRET] look_at', x, y, z)

        angle_xy = self.calculate_angle_xy(x, y)
        angle_yz = self.calculate_angle_yz(x, y, z)

        print(f'[TURRET] angle_xy: {angle_xy}, angle_yz: {angle_yz}')

        self.servo_xy.angle = angle_xy
        self.servo_yz.angle = angle_yz

    def calculate_angle_xy(self, x, y):
        """Calculate angle on the X-Y plane (azimuth)"""
        angle = math.degrees(math.atan2(y, x))  # ↑ = 90°, → = 0°
        if (angle<0):
            angle = -1 * angle
        # print(f'[TURRET] angle_xy: {angle}')
        if(y<0):
            angle = 90 + (90-angle) # -> 360-
        return angle

    def calculate_angle_yz(self, x, y, z):
        """Map laser elevation: close target = up (180), far = down (0)"""
        horizontal_dist = math.sqrt(x ** 2 + y ** 2)
        angle = math.degrees(math.atan2(z, horizontal_dist))  # ↑ = 90°, → = 0°
        return angle if y>=0 else 180 - angle

    def off(self):
        self.kit.servo[CHANNEL_SERVO_XY].angle = 0
        self.kit.servo[CHANNEL_SERVO_YZ].angle = 0
        print('[TURRET] off')
        self.laser.off()
