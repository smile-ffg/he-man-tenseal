from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.onnx import export as export_onnx

from he_man_tenseal.config import KeyParamsConfig
from he_man_tenseal.inference import ONNXModel

OPSET_VERSION = 14
np.random.seed(1)


class BatchNormModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        return x

    def export(self, filename: str, input: Union[torch.tensor, np.ndarray]) -> None:
        export_onnx(
            self,
            input,
            filename,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batchsize"}, "output": {0: "batchsize"}},
        )

        print("Model was saved as {}".format(filename))

    def to_string(self) -> str:
        pass

    def get_dim(self) -> int:
        pass


class BatchNorm1DModel(BatchNormModel):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(100)

    def to_string(self) -> str:
        return "BatchNorm1DModel"

    def get_dim(self) -> int:
        return 1


class BatchNorm2DModel(BatchNormModel):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(10)

    def to_string(self) -> str:
        return "BatchNorm2DModel"

    def get_dim(self) -> int:
        return 2


def eval_model(model: BatchNormModel, input: torch.tensor) -> None:
    # setup for models
    lr = 1  # high learning rate such that model parameters change more
    loss_fct = F.cross_entropy

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # pseudo-output for learning
    y = torch.rand(input.shape)
    # one training iteration
    pred = model(input)
    loss = loss_fct(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # save model
    model.eval()
    model.export(f"{model.to_string()}.onnx", input)
    # compute and save output
    output = model(input).detach().numpy().flatten()
    np.save(f"bn{model.get_dim()}d_output.npy", output)

    # compute onnx output
    model_path = Path(__file__).parent / f"{model.to_string()}.onnx"
    config = KeyParamsConfig(
        key_params_path="",  # won't be  used
        model_path=model_path,
        n_bits_fractional_precision=21,  # won't be  used
        calibration_data_path=f"bn{model.get_dim()}d_calib.npz",
        relu_mode="deg3",  # won't be used
        domain_mode="min-max",  # won't be used
    )
    heman_model = ONNXModel(path=model_path, key_params_config=config)
    # reshape because GemmWrappedOperator.execute -> else branch: requires
    # input to 2D with shape (batch_size, *)
    # heman_output = heman_model(input.reshape(input.shape[0], -1))[0].numpy().flatten()
    heman_output = heman_model(input)[0].numpy().flatten()

    # save output
    np.save(f"bn{model.get_dim()}d_heman_output.npy", heman_output)

    print(f"diffs < 1e-6: {np.allclose(output, heman_output, atol=1e-6)}")
    print(f"max error: {np.max(np.abs(output - heman_output))}")


def batchnorm_test() -> None:
    # create inputs for 1D and 2D
    one_d_input = torch.rand(20, 100)
    two_d_input = torch.randn(2, 10, 5, 5)

    np.save("bn1d_input.npy", one_d_input.numpy().flatten())
    np.save("bn2d_input.npy", two_d_input.numpy().flatten())

    # also use this as calibration-data later
    np.savez_compressed("bn1d_calib", one_d_input.numpy())
    np.savez_compressed("bn2d_calib", two_d_input.numpy())

    model = BatchNorm1DModel()
    print(f"Evaluate {model.to_string()}")
    eval_model(model, one_d_input)
    model = BatchNorm2DModel()
    print(f"Evaluate {model.to_string()}")
    eval_model(model, two_d_input)


if __name__ == "__main__":
    # tests the BatchNormalizationOperator implementation in inference.py
    #
    # 2 models are generated, one for BatchNorm1d, one for BatchNorm2d.
    # Models are trained 1 epoch with 1 batch to change initial BatchNorm parameters.
    # Output for random inputs are computed and compared against HE-MAN output.
    #
    batchnorm_test()
