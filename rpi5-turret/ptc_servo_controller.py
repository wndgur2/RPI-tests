import time
from adafruit_servokit import ServoKit
import adafruit_pca9685
import board
import busio


i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 100


kit = ServoKit(channels=16)

kit.servo[0].set_pulse_width_range(500, 2500)
kit.servo[1].set_pulse_width_range(500, 2500)

kit.servo[0].angle = 180
time.sleep(1)
kit.servo[0].angle = 150
time.sleep(1)
kit.servo[0].angle = 120
time.sleep(1)
kit.servo[0].angle = 90
time.sleep(1)
kit.servo[0].angle = 60
time.sleep(1)
kit.servo[0].angle = 30
time.sleep(1)
kit.servo[0].angle = 0


# kit.continuous_servo[1].throttle = 1
# time.sleep(1)
# kit.continuous_servo[1].throttle = -1
# time.sleep(1)
# kit.continuous_servo[1].throttle = 0