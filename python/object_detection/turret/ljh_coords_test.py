from turret import Turret
import time
import math

Z = 500
x=-250
y=-100

turret = Turret()
turret.laser.on()
time.sleep(1)

while(True):
    turret.look_at(x, y, Z)
    x += 10
    time.sleep(0.6)

time.sleep(100)
turret.off()