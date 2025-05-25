from adafruit_servokit import ServoKit
import adafruit_pca9685
import board
import busio
import time
import random
from gpiozero import LED

laser = LED(17)

i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 100

kit = ServoKit(channels=16)

# kit.servo[2]._pwm_out.duty_cycle = 0
# kit.servo[3]._pwm_out.duty_cycle = 0

kit.servo[0].set_pulse_width_range(500, 2500)
kit.servo[1].set_pulse_width_range(500, 2500)

kit.servo[1].angle = 90

laser.on()


for i in range(5):
    if(kit.servo[0].angle + 45 > 180):
        kit.servo[0].angle = (kit.servo[0].angle+45) % 180
    else:
        kit.servo[0].angle += 45

    time.sleep(1)

laser.off()

kit.servo[0].angle = 0
kit.servo[1].angle = 0


# kit.servo[2].angle = 180
# kit.servo[3].angle = 180

# time.sleep(2)

# kit.continuous_servo[0].throttle = 1
# time.sleep(2)
# kit.continuous_servo[0].throttle = -1
# time.sleep(2)
# kit.continuous_servo[1].throttle = 1
# time.sleep(2)
# kit.continuous_servo[1].throttle = -1
# time.sleep(2)