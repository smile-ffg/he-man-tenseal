from pathlib import Path
from typing import Optional, Sequence

import onnx
import torch
import torch.nn as nn
from onnx.helper import make_node, make_tensor_value_info
from torch.onnx import export as export_onnx

# make this script reproducible
torch.manual_seed(1337)

OPSET_VERSION = 14
Type = onnx.TensorProto.DataType

TEST_DIR = Path(__file__).parent / ".." / "tests"
MODELS_DIR = TEST_DIR / "models"
FIXED_DEPTH_MODELS_DIR = TEST_DIR / "models-fixed-depth"
APPROXIMATED_MODELS_DIR = TEST_DIR / "models-approximated"


def export(model: nn.Module, input_shape: list[int], filename: Path) -> None:
    if not filename.exists():
        export_onnx(
            model, torch.empty([1] + input_shape), filename, opset_version=OPSET_VERSION
        )
        print(f"exported {filename}")


# GEMM model
n_inputs = 16
n_outputs = 8
model = nn.Sequential(nn.Linear(n_inputs, n_outputs))
export(model, [n_inputs], MODELS_DIR / "gemm.onnx")

# matmul model
n_inputs = 16
n_outputs = 8
model = nn.Sequential(nn.Linear(n_inputs, n_outputs, bias=False))
export(model, [n_inputs], MODELS_DIR / "matmul.onnx")


# mul model
class MulModel(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.y = torch.rand(n_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.y


n_inputs = 16
model = MulModel(n_inputs)
export(model, [n_inputs], MODELS_DIR / "mul.onnx")


# add model
class AddModel(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.y = torch.rand(n_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.y


n_inputs = 16
model = AddModel(n_inputs)
export(model, [n_inputs], MODELS_DIR / "add.onnx")


# power 2 plus power 4 model
class Power2PlusPower4Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * x
        return x + x * x


n_inputs = 16
model = Power2PlusPower4Model()
export(model, [n_inputs], MODELS_DIR / "power2-plus-power4.onnx")


# fixed depth models that keep the value range fixed
class FixedDepthModel(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.depth):
            x = x * 1.0
        return x


for depth in [1, 2, 3]:
    model = FixedDepthModel(depth)
    export(model, [n_inputs], FIXED_DEPTH_MODELS_DIR / f"depth{depth}.onnx")


# conv model
n_inputs = 28 * 28
n_channels_in = 1
n_channels_out = 4
kernel_size = 7
stride = 3
model = nn.Conv2d(n_channels_in, n_channels_out, kernel_size, stride)
export(model, [1, 28, 28], MODELS_DIR / "conv.onnx")

# second conv model with non-default params
n_channels_in = 2
n_channels_out = 3
kernel_size_2d = (7, 6)
stride_2d = (5, 6)
pads = (2, 2)
dilation = (2, 3)
model = nn.Conv2d(
    n_channels_in,
    n_channels_out,
    kernel_size_2d,
    stride_2d,
    pads,
    dilation,
)
export(model, [n_channels_in, 28, 28], MODELS_DIR / "conv_v2.onnx")


model = nn.AvgPool1d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
input_shape = [1, 15]  # output_shape = [1, 8]
export(model, input_shape, MODELS_DIR / "avgpool1d.onnx")


model = nn.AvgPool2d(
    kernel_size=[2, 3], stride=[2, 1], padding=1, count_include_pad=False
)
input_shape = [1, 6, 9]  # output_shape = [1, 4, 9]
export(model, input_shape, MODELS_DIR / "avgpool2d.onnx")


model = nn.AvgPool3d(
    kernel_size=[2, 4, 3], stride=2, padding=[1, 2, 1], count_include_pad=False
)
input_shape = [1, 6, 8, 9]  # output_shape = [1, 4, 5, 5]
export(model, input_shape, MODELS_DIR / "avgpool3d.onnx")


# model that requires 3 bits for the integer part if the input domain is [-1, 1]
class ThreeBitIntModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1]
        y = 4 * x  # y in [-4, 4]
        z = y + (-3.5)  # z in [-7.5, 0.5]
        return 0.5 * z  # in [-3.75, 0.25]


n_inputs = 16
model = ThreeBitIntModel()
export(model, [n_inputs], MODELS_DIR / "3-bit-int-model.onnx")

n_inputs = 16
model = nn.ReLU()
export(model, [n_inputs], APPROXIMATED_MODELS_DIR / "relu.onnx")


def create_single_input_single_output_onnx_model(
    name: str,
    input_shape: Sequence[Optional[int]],
    output_shape: Sequence[Optional[int]],
    nodes: Sequence[onnx.onnx_ml_pb2.NodeProto],
    output_dir: Path = MODELS_DIR,
) -> None:
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            name,
            [make_tensor_value_info("input", Type.FLOAT, input_shape)],
            [make_tensor_value_info("output", Type.FLOAT, output_shape)],
        )
    )
    model.opset_import[0].version = OPSET_VERSION
    onnx.checker.check_model(model)

    output_path = output_dir / f"{name}.onnx"
    if not output_path.exists():
        onnx.save(model, output_path)
        print(f"exported {output_path}")


create_single_input_single_output_onnx_model(
    name="pad-zero-2d",
    input_shape=[1, 6, 7],
    output_shape=[1, 6 + 1 + 3, 7 + 4 + 2],
    nodes=[
        make_node("Constant", [], ["pads"], value_ints=[0, 1, 4, 0, 3, 2]),
        make_node("Pad", ["input", "pads"], ["output"], mode="constant"),
    ],
)

create_single_input_single_output_onnx_model(
    name="pad-constant-3d",
    input_shape=[1, 6, 7, 8],
    output_shape=[1, 6 + 1 + 4, 7 + 6 + 5, 8 + 3 + 2],
    nodes=[
        make_node("Constant", [], ["pads"], value_ints=[0, 1, 6, 3, 0, 4, 5, 2]),
        make_node("Constant", [], ["value"], value_float=1.0),
        make_node("Pad", ["input", "pads", "value"], ["output"], mode="constant"),
    ],
)

create_single_input_single_output_onnx_model(
    name="pad-reflection-1d",
    input_shape=[1, 7],
    output_shape=[1, 7 + 2 + 3],
    nodes=[
        make_node("Constant", [], ["pads"], value_ints=[0, 2, 0, 3]),
        make_node("Pad", ["input", "pads"], ["output"], mode="reflect"),
    ],
)

create_single_input_single_output_onnx_model(
    name="pad-edge-2d",
    input_shape=[1, 6, 7],
    output_shape=[1, 6 + 1 + 3, 7 + 4 + 2],
    nodes=[
        make_node("Constant", [], ["pads"], value_ints=[0, 1, 4, 0, 3, 2]),
        make_node("Pad", ["input", "pads"], ["output"], mode="edge"),
    ],
)
