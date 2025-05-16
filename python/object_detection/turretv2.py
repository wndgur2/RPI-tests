import math
import time
from adafruit_servokit import ServoKit
from gpiozero import LED

# 서보 채널
CHANNEL_SERVO_XY = 0
CHANNEL_SERVO_YZ = 1

# 서보 펄스폭
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# 레이저 위치 (기준점)
LASER_X = 0
LASER_Y = 0
LASER_Z = 0

# GPIO 핀
PIN_LASER = 27

# 보정값
OFFSET_Y = 151.5
OFFSET_Z = 30

class Turret:
    def __init__(self):
        self.kit = ServoKit(channels=16)

        # 서보 세팅
        self.servo_xy = self.kit.servo[CHANNEL_SERVO_XY]
        self.servo_yz = self.kit.servo[CHANNEL_SERVO_YZ]
        self.servo_xy.set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        self.servo_yz.set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)

        self.servo_xy.angle = 90
        self.servo_yz.angle = 90

        self.laser = LED(PIN_LASER)

    def look_at(self, x, y, z):
        y -= OFFSET_Y
        z -= OFFSET_Z

        print('[TURRET] look_at', x, y, z)

        angle_xy = self.calculate_angle_xy(x, y)
        angle_yz = self.calculate_angle_yz(x, y, z)

        print(f'[TURRET] angle_xy: {angle_xy:.2f}, angle_yz: {angle_yz:.2f}')

        self.servo_xy.angle = self.clamp(angle_xy)
        self.servo_yz.angle = self.clamp(angle_yz)

    def calculate_angle_xy(self, x, y):
        """XY 평면에서 회전 각도 계산 (수평 각도, azimuth)"""
        angle = math.degrees(math.atan2(y, x))  # atan2는 -180도~180도 반환
        angle = (angle + 360) % 360  # 0~360도로 정규화
        return angle if angle <= 180 else 360 - angle  # 0~180도로 변환

    def calculate_angle_yz(self, x, y, z):
        """레이저 상하 각도 계산 (elevation)"""
        horizontal_dist = math.sqrt(x**2 + y**2)
        angle = math.degrees(math.atan2(z, horizontal_dist))  # ↑=90°, →=0°
        return 90 - angle  # 서보 기준: 90=center

    def clamp(self, angle):
        """각도를 0~180 사이로 제한"""
        return max(0, min(180, angle))

    def off(self):
        self.servo_xy.angle = 90
        self.servo_yz.angle = 90
        self.laser.off()

def xTest(turret, y):
    target_x = -1000
    while target_x < 1000:
        turret.look_at(target_x, y, 500)
        target_x += 100
        time.sleep(0.5)

def yTest(turret, x):
    target_y = -1000
    while target_y < 1000:
        turret.look_at(x, target_y, 500)
        target_y += 100
        time.sleep(0.5)

if __name__ == '__main__':
    turret = Turret()
    try:
        # 테스트 사용 예
        # xTest(turret, 100)
        yTest(turret, 100)
    finally:
        turret.off()
