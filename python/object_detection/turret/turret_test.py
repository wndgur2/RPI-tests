from turret import Turret
import time

def xTest(turret, y):
    # X축 회전 테스트
    target_x = -1000
    while target_x<1000:
        turret.look_at(target_x, y, 500)
        target_x += 100
        time.sleep(0.3)

def yTest(turret, x):
    target_y = -1000
    while target_y<1000:
        turret.look_at(x, target_y, 500)
        target_y += 100
        time.sleep(0.3)

def diagonalTest(turret):
    # 대각선 회전 테스트
    target_x = -1000
    target_y = -1000
    while target_x<1000 and target_y<1000:
        turret.look_at(target_x, target_y, 500)
        target_x += 100
        target_y += 100
        time.sleep(0.3)

def zTest(turret, target_z):
    target_x = 0
    target_y = 0
    turret.look_at(target_x, target_y, target_z)

turret = Turret()

# xTest(turret, -300)
# yTest(turret, 0)
# diagonalTest(turret)

zTest(turret,740)


time.sleep(100)
turret.off()