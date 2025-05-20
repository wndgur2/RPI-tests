from turret import Turret
import time
import math

from correction_map import ServoAngleCorrector

corrector = ServoAngleCorrector("servo_calibration1.csv")


turret = Turret()
turret.laser.on()
# time.sleep(1)

turret.servos[0].angle = 45
turret.servos[1].angle = 0

angle=0

while(True):
    if(angle>100):
        angle=45
    elif(angle<100):
        angle=135
    turret.servos[0].angle = angle
    time.sleep(3)

# while(angle<=180):
#     corrected_angle = corrector.correct(angle)
#     print(f"Desired: {angle}, Corrected: {corrected_angle}")

#     if(corrected_angle>180):
#         corrected_angle = 180

#     turret.servos[0].angle = corrected_angle
#     angle += 20
#     time.sleep(2)
#     print(f"Servo 0 angle: {turret.servos[0].angle}")
#     print('--------------------')

turret.off()