from pathlib import Path
import numpy as np


def load_calibration_data(path: Path) -> np.ndarray:
    calibration_data = np.load(path)
    calibration_data = np.array(
        [calibration_data[file] for file in calibration_data.files]
    )
    calibration_data = calibration_data.reshape(-1, *calibration_data.shape[2:])
    return [calibration_data]
