from turret import Turret
import time

Z = 820

DELAY = 0.2

def xTest(turret, y):
    # X축 회전 테스트
    target_x = -250
    while target_x<300:
        turret.look_at(target_x, y, Z)
        target_x += 10
        time.sleep(DELAY)

def yTest(turret, x):
    target_y = 250
    while target_y>-450:
        turret.look_at(x, target_y, Z)
        target_y -= 20
        time.sleep(DELAY)

def diagonalTest(turret):
    # 대각선 회전 테스트
    target_x = -1000
    target_y = -1000
    while target_x<1000 and target_y<1000:
        turret.look_at(target_x, target_y, Z)
        target_x += 100
        target_y += 100
        time.sleep(DELAY)

def zTest(turret, target_z):
    target_x = 0
    target_y = 0
    turret.look_at(target_x, target_y, target_z)

def angleTest(turret):
    angle_xy = 0
    angle_yz = 90
    for x in range(5):
        angle_xy +=45
        if(angle_xy > 180):
            angle_xy %= 180
            angle_yz = 90

        # angle_yz +=1
        # angle_xy +=-0.4
        # if(angle_yz>105):
            # angle_yz =85
            # angle_xy =90
        turret.servo_xy.angle = angle_xy
        turret.servo_yz.angle = angle_yz
        print(angle_xy)
        time.sleep(1)

turret = Turret()
turret.laser.on()
# turret.look_at(0,0,Z)
# time.sleep(20)``

# yTest(turret, 10)
# xTest(turret, 150)

angleTest(turret)
# diagonalTest(turret)
# turret.off()

turret.off()