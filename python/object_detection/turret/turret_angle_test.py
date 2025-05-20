from turret2 import Turret
import time

turret = Turret()

turret.laser.on()
xx = 0
yy = 0
zz = 700


angle_xy = 0
angle_yz = 90

# angle_xy = 90
# angle_yz = 85

while True:
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
    time.sleep(1)


# while True:
#     turret.look_at(xx,yy,zz)
    # time.sleep(10)
#     turret.look_at(xx+100,yy,zz)
#     time.sleep(10)
#     turret.look_at(xx+200,yy,zz)
#     time.sleep(10)
#     turret.look_at(xx+300,yy,zz)
#     time.sleep(10)

# while True:
#     turret.look_at(xx,yy,zz)
#     time.sleep(10)
#     turret.look_at(xx,yy+50,zz)
#     time.sleep(10)
#     turret.look_at(xx,yy+100,zz)
#     time.sleep(10)
#     turret.look_at(xx,yy+150,zz)
#     time.sleep(10)



# turret.look_at(0,0,760)
# zTest(turret,99999)
# turret.laser.on()

turret.off()