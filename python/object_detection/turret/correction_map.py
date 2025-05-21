import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ServoAngleCorrector:
    def __init__(self, calibration_csv_path):
        df = pd.read_csv(calibration_csv_path)
        target = df["target_angle"].values
        measured = df["measured_angle"].values

        # remove duplicated x
        measured, indices = np.unique(measured, return_index=True)
        target = target[indices]

        # Build inverse map: from measured → target
        self.correct_fn = interp1d(
            measured, target,
            kind='cubic',  # or 'quadratic' or 'linear'
            fill_value="extrapolate",
            bounds_error=False
        )

    def correct(self, desired_angle):
        measured_angle = self.correct_fn(desired_angle)

        print(f"[CORRECTOR] Correcting angle: {desired_angle} to {measured_angle}")
        return measured_angle

    def plot(self):
        x = np.linspace(0, 180, 500)
        y = self.correct(x)
        plt.plot(x, y, label="Correction Function")
        plt.xlabel("Desired Output Angle")
        plt.ylabel("Corrected Input Angle")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig("correction_map.png")
        print("Saved correction map to correction_map.png")
