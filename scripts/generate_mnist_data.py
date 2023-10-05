import argparse
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def generate_mnist_calibration_data(n_samples: int = 10000) -> None:
    transformation = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

    data_dir = Path(__file__).parent / ".." / "data"
    train_data = datasets.MNIST(
        data_dir, train=True, download=True, transform=transformation
    )
    train_loader = DataLoader(train_data, batch_size=n_samples, shuffle=True)
    train_dataset_array = next(iter(train_loader))[0].numpy()
    calibration_data = train_dataset_array.reshape(-1, 1, 32, 32).astype(np.float32)

    calibration_data_path = Path(__file__).parent / ".." / "demo" / "mnist"
    filename = calibration_data_path / "mnist_32x32"

    if not filename.exists():
        np.savez_compressed(filename, calibration_data)


def save_demo_sample() -> None:
    transformation = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    data_dir = Path(__file__).parent / ".." / "data"
    test_data = datasets.MNIST(
        data_dir, train=False, download=True, transform=transformation
    )
    test_loader = DataLoader(test_data, batch_size=1)
    sample = next(iter(test_loader))[0].numpy().astype(np.float32)

    path = Path(__file__).parent / ".." / "demo" / "mnist" / "input.npy"
    np.save(path, sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HE-MAN evaluation")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="calibration",
        choices=["calibration", "sample"],
        help="Do you want to generate calibration data (.npz) "
        "or an input sample (.npy)?",
    )

    args = parser.parse_args()
    if args.method == "calibration":
        generate_mnist_calibration_data()
    elif args.method == "sample":
        save_demo_sample()
    else:
        print("Invalid method parameter!")
