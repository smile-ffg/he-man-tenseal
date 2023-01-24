from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.onnx import export as export_onnx
from torch.utils.data import DataLoader
from torchvision import datasets

torch.manual_seed(1337)
OPSET_VERSION = 14


class MnistSquareModel(nn.Module):
    def __init__(self, input_shape: tuple = (28, 28), hidden: int = 30) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=5, padding=0, stride=3
        )
        self.fc1 = nn.Linear(400, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # convolutional layer
        x = self.conv1(x)
        # square activation
        x = x * x
        # flatten to [batch_size, 256]
        x = torch.flatten(x, start_dim=1)
        # fc1 layer
        x = self.fc1(x)
        # square activation
        x = x * x
        # fc2 layer
        x = self.fc2(x)
        return x

    def export_as_onnx(self) -> None:
        output_dir = Path(__file__).parent / ".." / "demo" / "mnist"
        export_onnx(
            self,
            torch.empty([1, 1] + list(self.input_shape)),
            output_dir / "cryptonets_32x32.onnx",
            opset_version=OPSET_VERSION,
        )


def get_data(batch_size: int, data_shape: tuple = (28, 28)) -> tuple:
    """
    Returns tuple with three DataLoaders for training, validation and test data.
    MNIST data is taken from ../data/mnist or downloaded if not available.
    Subdirectory mnist is created automatically.
    """
    transformation = transforms.Compose(
        [transforms.Resize(data_shape), transforms.ToTensor()]
    )

    data_dir = Path(__file__).parent / ".." / "data"
    train_data = datasets.MNIST(
        data_dir, train=True, download=True, transform=transformation
    )
    train_ds, valid_ds = torch.utils.data.random_split(train_data, [50000, 10000])
    test_ds = datasets.MNIST(
        data_dir, train=False, download=False, transform=transformation
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size * 2),
        DataLoader(test_ds, batch_size=1),
    )


def loss_batch(
    model: MnistSquareModel,
    loss_func: torch.nn.functional,
    xb: torch.Tensor,
    yb: torch.Tensor,
    opt: optim.SGD = None,
) -> tuple:
    """
    Returns loss and number of prediction matches for outside accuracy calculation.
    """
    y_pred = model(xb)
    loss = loss_func(y_pred, yb)

    # accuracy: correct classifications
    matches = (torch.argmax(y_pred, dim=1) == yb).float().sum()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), matches


def train(
    model: MnistSquareModel,
    epochs: int,
    loss_func: torch.nn.functional,
    optimizer: optim.SGD,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> MnistSquareModel:
    print("  |   TRAINING    |     TEST")
    print("ep|  loss   acc   |  loss   acc")
    print("---------------------------------")
    for e in range(1, epochs + 1):
        model.train()
        losses, matches = zip(
            *[loss_batch(model, loss_func, xb, yb, optimizer) for xb, yb in train_dl]
        )
        train_loss = np.sum(losses) / len(train_dl)
        train_acc = np.sum(matches) / len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            losses, matches = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )

        eval_loss = np.sum(losses) / len(valid_dl)
        eval_acc = np.sum(matches) / len(valid_dl.dataset)

        print(
            f"{e:2d}| {train_loss:.4f} {train_acc:.4f} | {eval_loss:.4f} {eval_acc:.4f}"
        )

    return model


def train_demo_model() -> None:
    loss_func = F.cross_entropy
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 15
    batch_size = 32

    model = MnistSquareModel(input_shape=(32, 32), hidden=128)
    train_dl, valid_dl, test_dl = get_data(batch_size=batch_size, data_shape=(32, 32))
    opt = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model = train(
        model,
        epochs=epochs,
        loss_func=loss_func,
        optimizer=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
    )

    model.export_as_onnx()


if __name__ == "__main__":
    train_demo_model()
