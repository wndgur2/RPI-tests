from turret import Turret
import time

Z = 820

DELAY = 0.2

def xTest(turret, y):
    # X축 회전 테스트
    target_x = -250
    while target_x<500:
        turret.look_at(target_x, y, Z)
        target_x += 10
        time.sleep(DELAY)

def yTest(turret, x):
    target_y = 50
    while target_y>-350:
        turret.look_at(x, target_y, Z)
        target_y -= 10
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

turret = Turret()
turret.laser.on()
turret.look_at(0,0,Z)
time.sleep(20)

# yTest(turret, 10)
# xTest(turret, -50)
# diagonalTest(turret)

turret.off()