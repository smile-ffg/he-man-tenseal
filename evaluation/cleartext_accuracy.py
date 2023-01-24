from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from tenseal_inference.config import KeyParamsConfig
from tenseal_inference.inference import ONNXModel

EVAL_DIR = Path(__file__).parent
MODEL_PATH = EVAL_DIR / "lenet5.onnx"
N_SAMPLES = 1000

#
#  ATTENTION
#  Cleartext accuracy is only working if ReLU operators
#  use exact ReLU and not a polynomial approximation.
#
#  So far, this has to be changed in the code.
#  To be implemented: flag denoting a calibrated model.
#


def get_data(data_shape: tuple = (32, 32)) -> tuple:
    transformation = transforms.Compose(
        [transforms.Resize(data_shape), transforms.ToTensor()]
    )
    data_dir = Path(__file__).parent / ".." / "data"
    test_data = datasets.MNIST(
        data_dir, train=False, download=True, transform=transformation
    )
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    test_dataset_array = next(iter(test_loader))[0].numpy()
    data = test_dataset_array.reshape(10000, 1, 32, 32).astype(np.float32)

    labels = test_data.targets.numpy()

    return (data, labels)


def execute_inference(
    model_path: Path,
    calibration_data_path: Path,
    input_data: np.array,
    correct_labels: np.array,
) -> int:
    config = KeyParamsConfig(
        key_params_path="",  # won't be  used
        model_path=model_path,
        n_bits_fractional_precision=21,  # represents a minimum
        calibration_data_path=calibration_data_path,
        relu_mode="deg3",
        domain_mode="min-max",
    )

    model = ONNXModel(path=model_path, key_params_config=config)

    # perform inference
    output = model(input_data)[0]
    predictions = np.argmax(output, axis=1)

    return (predictions == correct_labels).astype(float).mean()


def evaluate_mnist_models() -> None:
    models = ["cryptonets", "lenet5"]
    calibration_data_path = EVAL_DIR / "mnist_calibration-data.zip"

    # prepare inputs
    input_data, correct_labels = get_data()

    for model in models:
        model_path = EVAL_DIR / (model + ".onnx")

        accuracy = execute_inference(
            model_path, calibration_data_path, input_data, correct_labels
        )

        print(f"{model} {accuracy:.3f}")


def evaluate_facenet_model() -> None:
    pass


def perform_cleartext_evaluation() -> None:
    evaluate_mnist_models()
    evaluate_facenet_model()


if __name__ == "__main__":
    perform_cleartext_evaluation()
