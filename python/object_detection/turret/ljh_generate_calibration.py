from turret import Turret
import time
import csv

CHANNEL = 0

turret = Turret()
turret.laser.off()

# Set initial position
turret.servos[CHANNEL].angle = 0
time.sleep(1)

# Open a CSV file to write
with open(f"servo_calibration{CHANNEL}.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["target_angle", "measured_angle"])

    start_angle = 0
    while(start_angle<0.5):
        target_angle = start_angle
        time.sleep(1)
        while (target_angle<=180):

            actual_angle = turret.servos[CHANNEL].angle
            print(f"Target: {target_angle}, Measured: {actual_angle}")
            writer.writerow([target_angle, actual_angle])

            turret.servos[CHANNEL].angle = target_angle
            time.sleep(0.1)  # wait for servo to settle

            target_angle += 0.5
        start_angle += 0.1