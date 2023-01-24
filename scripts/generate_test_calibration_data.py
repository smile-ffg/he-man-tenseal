from pathlib import Path

import numpy as np


def generate_and_save_calibration_data(lower: int, upper: int) -> None:
    lower = int(round(lower))
    upper = int(round(upper))

    if lower > upper:
        tmp = lower
        lower = upper
        upper = tmp

    calibration_data_path = Path(__file__).parent / ".." / "tests" / "calibration-data"
    filename = calibration_data_path / f"lower_{lower}_upper_{upper}"

    if not filename.exists():
        np.savez_compressed(filename, np.array([lower, upper]))


bounds = [
    # test_keyparams.py
    (0, 100),
    (2000, 2000),
    (-512, 0),
    # test_encrypt_decrypt.py & test_inference.py
    (0, 1000),
    # test_inference.py -> test_int_precision
    (-1, 1),
    # test_inference.py
    (0, 5),
    (0, 200),
]

for lower, upper in bounds:
    generate_and_save_calibration_data(lower, upper)
