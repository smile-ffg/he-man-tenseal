from pathlib import Path
from typing import Dict

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import tenseal_inference.crypto as crypto
from tenseal_inference.config import KeyParamsConfig
from tenseal_inference.inference import ONNXModel

N_SAMPLES = 10000
EVAL_DIR = Path(__file__).parent
degrees = [1, 3, 7]


def evaluate_relu_approximations() -> None:
    results: Dict[str, Dict] = dict()

    transformation = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    data_dir = Path(__file__).parent / ".." / "data"
    test_data = datasets.MNIST(
        data_dir, train=False, download=True, transform=transformation
    )
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    test_dataset_array = next(iter(test_loader))[0].numpy()
    input_data = test_dataset_array.reshape(10000, 1, 32, 32).astype(np.float32)

    correct_labels = test_data.targets.numpy()

    for degree in degrees:
        relu_mode = "deg" + str(degree)
        table_name = "OLS" + str(degree)
        results[table_name] = dict()

        for calibration in ["min-max", "mean-std"]:
            model_path = EVAL_DIR / "lenet5.onnx"

            # get crypto parameters
            config = KeyParamsConfig(
                key_params_path="",  # won't be  used
                model_path=model_path,
                n_bits_fractional_precision=21,  # represents a minimum
                calibration_data_path=EVAL_DIR / "mnist_calibration-data.zip",
                relu_mode=relu_mode,
                domain_mode=calibration,
            )

            # calibrate model
            model = ONNXModel(path=model_path, key_params_config=config)

            key_params = crypto.find_optimal_parameters(config, model)

            # perform inference
            output = model(input_data)[0]
            predictions = np.argmax(output, axis=1)

            accuracy = (predictions == correct_labels).astype(float).mean()

            d_m = (len(key_params.coeff_mod_bit_sizes) - 2 - 4) // 4

            results[table_name][calibration] = accuracy
            # d_m and log2N stay the same across calibration method
            results[table_name]["d_m"] = d_m
            results[table_name]["log2N"] = int(np.log2(key_params.poly_modulus_degree))

    # print results: Table 3 in the paper
    print(f"{' '*7} | {'Calibration':^21s} |     |")
    print(
        f"{'Method':<7s} | {'min - max':^9s} | {'u +- 3s':^9s} | d_m | {'log2 N':^8s}"
    )
    print("-" * 47)
    for poly, metrics in results.items():
        print(
            f"{poly:<7s} | {metrics['min-max']:^9.3f} | {metrics['mean-std']:^9.3f} | \
                {metrics['d_m']:^3d} | {metrics['log2N']:^6d}"
        )


if __name__ == "__main__":
    evaluate_relu_approximations()
