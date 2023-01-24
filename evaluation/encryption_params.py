import shutil
from pathlib import Path

import numpy as np

import he_man_tenseal.crypto as crypto
from he_man_tenseal.config import KeyParamsConfig
from he_man_tenseal.inference import ONNXModel

EVAL_DIR = Path(__file__).parent


def derive_parameters(calibration_data_path: Path, model_path: Path) -> list:
    config = KeyParamsConfig(
        key_params_path="",  # won't be  used
        model_path=model_path,
        n_bits_fractional_precision=21,  # represents a minimum
        calibration_data_path=calibration_data_path,
        relu_mode="deg3",
        domain_mode="min-max",
    )

    model = ONNXModel(path=model_path, key_params_config=config)

    key_params = crypto.find_optimal_parameters(config, model)
    return [
        int(np.log2(key_params.poly_modulus_degree)),
        int(sum(key_params.coeff_mod_bit_sizes)),
        len(key_params.coeff_mod_bit_sizes) - 2,
    ]


def evaluate_mnist_models() -> dict[str, list]:
    models = ["cryptonets", "lenet5"]
    calibration_data_path = EVAL_DIR / "mnist_calibration-data.zip"

    results = {}
    for model in models:
        model_path = EVAL_DIR / (model + ".onnx")
        params = derive_parameters(calibration_data_path, model_path)
        results[model] = params

    return results


def evaluate_facenet_model() -> dict[str, list]:
    model_path = EVAL_DIR / "mobilefacenets.onnx"

    # check if zip file exists, if not -> zip the calibration data
    calibration_data_path = EVAL_DIR / "lfw_calibration-data.zip"
    if not calibration_data_path.exists():
        if (EVAL_DIR / "lfw_calibration-data").exists():
            shutil.make_archive(
                str(calibration_data_path.parent / calibration_data_path.stem),
                "zip",
                root_dir="lfw_calibration-data/",
                base_dir="./",
            )
        else:
            print("No LFW calibration data found.")
            return {}

    params = derive_parameters(calibration_data_path, model_path)
    return {"mobilefacenets": params}


def evaluate_encryption_parameters() -> None:
    mnist_results = evaluate_mnist_models()
    facenet_results = evaluate_facenet_model()

    print(f"{'Network':<17s} {'log2 N':^6s}  {'log2 Q':^6s}  {'d_m':3s}")
    print("-" * 38)
    for net, data in (mnist_results | facenet_results).items():
        print(f"{net:<17s} {data[0]:^6d}  {data[1]:^6d}  {data[2]:^3d}")


if __name__ == "__main__":
    # produces Table 5 in the paper
    evaluate_encryption_parameters()
