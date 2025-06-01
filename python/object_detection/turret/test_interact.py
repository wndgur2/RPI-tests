from turret import Turret
import time

Z = 820
DELAY = 0.2
STEP = 10  # 10mm per step

def xTest(turret, y):
    target_x = -250
    while target_x < 300:
        turret.look_at(target_x, y, Z)
        target_x += 10
        time.sleep(DELAY)

def yTest(turret, x):
    target_y = 250
    while target_y > -450:
        turret.look_at(x, target_y, Z)
        target_y -= 20
        time.sleep(DELAY)

def angleTest(turret):
    angle_xy = 0
    angle_yz = 90
    for _ in range(5):
        angle_xy += 45
        if angle_xy > 180:
            angle_xy %= 180
            angle_yz = 90
        turret.servo_xy.angle = angle_xy
        turret.servo_yz.angle = angle_yz
        print(f"angle_xy: {angle_xy}")
        time.sleep(1)

# Initial state
x, y = 0, 0
turret = Turret()
turret.laser.on()
turret.look_at(x, y, Z)

print("Commands: w/a/s/d to move, x = xTest, y = yTest, angle = angleTest, q = quit")

try:
    while True:
        print(f"\nCurrent Position: x={x}, y={y}")
        cmd = input("Enter command: ").strip().lower()

        if cmd == 's':
            y += STEP
        elif cmd == 'w':
            y -= STEP
        elif cmd == 'a':
            x -= STEP
        elif cmd == 'd':
            x += STEP
        elif cmd == 'x':
            xTest(turret, y)
        elif cmd == 'y':
            yTest(turret, x)
        elif cmd == 'angle':
            angleTest(turret)
        elif cmd == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid command. Use w/a/s/d/x/y/angle/q.")
            continue

        turret.look_at(x, y, Z)
        time.sleep(DELAY)

finally:
    turret.off()
