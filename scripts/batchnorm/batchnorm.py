from pathlib import Path
from typing import Union

import numpy as np
import tenseal as ts
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
        self.bn = nn.BatchNorm1d(1000)

    def to_string(self) -> str:
        return "BatchNorm1DModel"

    def get_dim(self) -> int:
        return 1


class BatchNorm2DModel(BatchNormModel):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(100)

    def to_string(self) -> str:
        return "BatchNorm2DModel"

    def get_dim(self) -> int:
        return 2


def eval_model(model: BatchNormModel, input: torch.tensor) -> None:
    # setup for models
    lr = 1  # high learning rate such that model parameters change more
    loss_fct = F.cross_entropy

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # training BN requires at least 2 batches
    training_input = torch.cat((input, input))
    # pseudo-output for learning
    y = torch.rand(training_input.shape)
    # one training iteration
    pred = model(training_input)
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
    # plaintext case
    heman_pt_output = heman_model(input)[0].numpy().flatten()

    # save output
    np.save(f"bn{model.get_dim()}d_heman_output.npy", heman_pt_output)

    print(f"diffs < 1e-6: {np.allclose(output, heman_pt_output, atol=1e-6)}")
    print(f"max plaintext error: {np.max(np.abs(output - heman_pt_output))}")

    # ciphertext case
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[55, 50, 50, 55],
    )
    context.generate_galois_keys()
    context.global_scale = 2**50

    enc_input = ts.ckks_vector(context, input.ravel())

    heman_ct_output = np.array(heman_model(enc_input)[0].decrypt())

    print(f"max ciphertext error: {np.max(np.abs(output - heman_ct_output))}")


def batchnorm_test() -> None:
    # create inputs for 1D and 2D
    one_d_input = torch.rand(1, 1000)
    two_d_input = torch.randn(1, 100, 5, 5)

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
