import io
import json
import time
from dataclasses import dataclass
from math import floor, log2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime
import tenseal as ts
from jsonschema import validate
from loguru import logger
from onnx import TensorProto, helper, numpy_helper
from onnx.onnx_pb import StringStringEntryProto

from tenseal_inference.config import KeyParamsConfig

OPSET_VERSION = 14


@dataclass
class Interval:
    lower_bound: float
    upper_bound: float

    def scale(self, s: float) -> "Interval":
        return Interval(self.lower_bound * s, self.upper_bound * s)

    @staticmethod
    def from_array(a: np.ndarray) -> "Interval":
        return Interval(np.min(a), np.max(a))

    @staticmethod
    def envelope(a: "Interval", b: "Interval") -> "Interval":
        return Interval(
            min(a.lower_bound, b.lower_bound), max(a.upper_bound, b.upper_bound)
        )

    @staticmethod
    def addition_domain(a: "Interval", b: "Interval") -> "Interval":
        return Interval(a.lower_bound + b.lower_bound, a.upper_bound + b.upper_bound)

    @staticmethod
    def multiplication_domain(a: "Interval", b: "Interval") -> "Interval":
        edge_values = [
            a.lower_bound * b.lower_bound,
            a.lower_bound * b.upper_bound,
            a.upper_bound * b.lower_bound,
            a.upper_bound * b.upper_bound,
        ]
        return Interval(min(edge_values), max(edge_values))


@dataclass
class TensorMetaInfo:
    multiplication_depth: int
    shape: List[int]
    dtype: int
    can_be_encrypted: bool
    domain: Optional[Interval] = None


class ONNXModel:
    def __init__(
        self,
        path: Path,
        key_params_config: Optional[KeyParamsConfig] = None,
        domain_factors: tuple = (1.0, 1.0),
    ):
        self._domain_factors = domain_factors
        self._model = onnx.load(path)
        self._path = path
        self._n_inputs = len(self._model.graph.input)
        self._initializer_state = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in self.initializers
        }

        self.relu_mode: Optional[str] = ""
        if key_params_config is not None:
            self._set_relu_mode(key_params_config.relu_mode)
        else:
            # inference case
            self._load_relu_mode()

        self.domain_min = 0
        self.domain_max = 0

        if key_params_config is not None:
            # bring calibration-data into a single batch structure
            calibration_data = np.load(key_params_config.calibration_data_path)

            calibration_data = np.array(
                [calibration_data[file] for file in calibration_data.files]
            )
            calibration_data = calibration_data.reshape(-1, *calibration_data.shape[2:])

        meta_info_initializers = {
            initializer.name: TensorMetaInfo(
                multiplication_depth=0,
                shape=list(initializer.dims),
                dtype=initializer.data_type,
                can_be_encrypted=False,
            )
            for initializer in self.initializers
        }
        meta_info_inputs = {
            input.name: TensorMetaInfo(
                multiplication_depth=0,
                shape=[
                    # note that there can be named dimensions without a dim_value,
                    # which are commonly used for batch indices
                    (1 if d.dim_param != "" else d.dim_value)
                    for d in input.type.tensor_type.shape.dim
                ],
                dtype=input.type.tensor_type.elem_type,
                can_be_encrypted=True,
            )
            for input in self.inputs
        }
        self.meta_info = meta_info_initializers | meta_info_inputs

        self._const_state: Dict[str, np.ndarray] = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in self.initializers
        }

        self._operators = [self._create_operator(node) for node in self.nodes]

        if key_params_config is not None:
            self._calibrate(calibration_data, key_params_config.domain_mode)
        else:
            self._load_domains_from_onnx_metadata()

    def _create_operator(self, node: onnx.onnx_ml_pb2.NodeProto) -> "Operator":
        try:
            logger.debug(f"create operator for node {node.name}")
            operator_class = getattr(self, f"{node.op_type}Operator")
        except AttributeError:
            raise NotImplementedError(f"ONNX operator '{node.op_type}' not implemented")

        operator = operator_class(self, node)

        if not isinstance(self.meta_info[operator.output].shape, list):
            raise TypeError(
                f"type of output shape of node {node.name} ({node.op_type}) is "
                f"{type(self.meta_info[operator.output].shape).__name__}, but should "
                "be list"
            )

        return operator

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def inputs(self) -> List[onnx.onnx_ml_pb2.ValueInfoProto]:
        return self._model.graph.input

    @property
    def outputs(self) -> List[onnx.onnx_ml_pb2.ValueInfoProto]:
        return self._model.graph.output

    @property
    def nodes(self) -> List[onnx.onnx_ml_pb2.NodeProto]:
        return self._model.graph.node

    @property
    def initializers(self) -> List[onnx.onnx_ml_pb2.TensorProto]:
        return self._model.graph.initializer

    @property
    def max_encrypted_size(self) -> int:
        return int(
            max(
                (
                    np.prod(meta_info.shape)
                    for meta_info in self.meta_info.values()
                    if meta_info.can_be_encrypted
                )
            )
        )

    def __call__(
        self, *inputs: List[Union[ts.CKKSVector, np.ndarray]]
    ) -> List[Union[ts.CKKSVector, np.ndarray]]:
        state = self._forward(*inputs)
        return [state[output.name] for output in self.outputs]

    def _forward(
        self, *inputs: List[Union[ts.CKKSVector, np.ndarray]]
    ) -> Dict[str, Union[ts.CKKSVector, np.ndarray]]:
        if len(inputs) != self.n_inputs:
            raise ValueError(
                f"Invalid number of inputs (expected {self.n_inputs}, but got "
                f"{len(inputs)})"
            )

        state = self._initializer_state | {
            input.name: a for input, a in zip(self.inputs, inputs)
        }

        for operator in self._operators:
            operator_start_time = time.time()
            operator.execute(state)
            operator_execution_time = time.time() - operator_start_time
            logger.info(
                f"execution of {operator.name} ({type(operator).__name__}) took "
                f"{operator_execution_time:.2f} s"
            )

        return state

    def get_state(
        self, *inputs: List[Union[ts.CKKSVector, np.ndarray]]
    ) -> Dict[str, Union[ts.CKKSVector, np.ndarray]]:
        state = self._forward(*inputs)
        result = {
            name: value
            for name, value in state.items()
            if self.meta_info[name].can_be_encrypted
        }
        return result

    def _calibrate(self, calibration_data: np.ndarray, method: str) -> None:
        # forward pass using calibration data
        state = self._forward(calibration_data)

        if method == "mean-std":
            domains = {
                name: {
                    "lower": float(np.mean(values) - 3 * np.std(values)),
                    "upper": float(np.mean(values) + 3 * np.std(values)),
                }
                for name, values in state.items()
            }
        elif method == "min-max":
            domains = {
                name: {"lower": float(np.amin(values)), "upper": float(np.amax(values))}
                for name, values in state.items()
            }
        else:
            raise ValueError("Invalid domain calibration method")

        for name, domain in domains.items():
            self.meta_info[name].domain = Interval(domain["lower"], domain["upper"])

        domain_props = [p for p in self._model.metadata_props if p.key == "domain"]
        # delete existing properties (potentially more than 1)
        for prop in domain_props:
            self._model.metadata_props.remove(prop)

        self._model.metadata_props.append(
            StringStringEntryProto(key="domain", value=json.dumps(domains))
        )

        self.domain_min = min([np.amin(v) for v in state.values()])
        self.domain_max = max([np.amax(v) for v in state.values()])

    def _load_domains_from_onnx_metadata(self) -> None:
        # if no calibration data -> look for metadata_props in onnx model
        domain_props = [p for p in self._model.metadata_props if p.key == "domain"]
        # check for one unique "domain" key
        if len(domain_props) == 1:
            domains = json.loads(domain_props[0].value)

            # check the json schema
            schema = {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "lower": {"type": "number"},
                        "upper": {"type": "number"},
                    },
                    "required": ["lower", "upper"],
                    "additionalProperties": False,
                },
            }

            try:
                validate(domains, schema)
            except Exception:
                raise ValueError("Invalid ONNX metadata_props format for key 'domain'")

            # check if all model edges are in the metadata
            if not all(edge in domains.keys() for edge in self.meta_info.keys()):
                raise ValueError(
                    "ONNX metadata does not contain domain values for all model edges"
                )

            # iterate over edges and save into meta_info-domains
            for edge in self.meta_info.keys():
                lower = domains[edge]["lower"]
                upper = domains[edge]["upper"]

                if lower > upper:
                    raise ValueError(
                        f"Invalid domain (min: {lower}, max: {upper}) in "
                        f"ONNX metadata_props for key {edge}."
                    )

                self.meta_info[edge].domain = Interval(lower, upper)

        elif len(domain_props) > 1:
            raise ValueError("Model includes multiple metadata_props with key 'domain'")

    def _set_relu_mode(self, relu_mode: str) -> None:
        # delete existing relu metadata properties (potentially more than 1)
        relu_props = [p for p in self._model.metadata_props if p.key == "relu_mode"]
        for prop in relu_props:
            self._model.metadata_props.remove(prop)

        self._model.metadata_props.append(
            StringStringEntryProto(key="relu_mode", value=relu_mode)
        )

        self.relu_mode = relu_mode

    def _load_relu_mode(self) -> None:
        relu_props = [p for p in self._model.metadata_props if p.key == "relu_mode"]

        if len(relu_props) == 1:
            self.relu_mode = relu_props[0].value
        elif len(relu_props) > 1:
            raise ValueError(
                "Model includes multiple metadata_props with key 'relu_mode'"
            )
        else:
            self.relu_mode = None

    def save_calibrated_model(self, suffix: str = "") -> None:
        calibrated_model_path = self._path.parent / (
            self._path.stem + "_calibrated" + suffix + self._path.suffix
        )
        if calibrated_model_path.exists():
            raise FileExistsError(
                f"There is already a file named {calibrated_model_path.name} "
                f"in {calibrated_model_path.parent}"
            )

        onnx.save(self._model, calibrated_model_path)

    @property
    def multiplication_depth(self) -> int:
        return max(
            self.meta_info[output.name].multiplication_depth for output in self.outputs
        )

    @property
    def n_bits_integer_precision(self) -> int:
        max_abs_value = max(abs(self.domain_min), abs(self.domain_max))
        return int(np.log2(max_abs_value) + 1)

    class Operator:
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            self.model = model
            self.node = node
            self.meta_info_inputs = [
                self.model.meta_info[input] for input in self.inputs
            ]
            self.attributes = {a.name: a for a in node.attribute}

        @property
        def name(self) -> str:
            return self.node.name

        @property
        def inputs(self) -> List[str]:
            return self.node.input

        @property
        def output(self) -> str:
            return self.node.output[0]

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            raise NotImplementedError("TODO: Implement in derived operator classes")

    class GemmWrappedOperator(Operator):
        """This abstract `GemmWrapperOperator` wraps an ONNX node into a GEMM
        operation. Although this can be much more expensive compared to directly
        evaluating the node, this approach is required for the application of some ONNX
        operators to CKKS vectors. Note that this approach only works for operators
        that are linear operations, e.g. convolutioal layers or average pooling.

        To implement a specific operator based on this class, simply create a subclass
        of this class and override the property `domain`. Note that only a single input
        may be encrypted and this input has be the input to the linear operation. If
        this input to the linear operation is not the first input, the subclass has to
        override the property `gemm_input_index`.
        """

        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            self.input_shape = self.meta_info_inputs[self.gemm_input_index].shape
            self.init_node_inference_session()
            self.init_meta_info()

        @property
        def gemm_input_index(self) -> int:
            """This index defines, which of the inputs to use as the GEMM input.

            Returns:
                int: The position index of the GEMM input.
            """
            return 0

        def init_node_inference_session(self) -> None:
            graph_inputs = [
                helper.make_tensor_value_info(
                    input_name,
                    self.meta_info_inputs[j].dtype,
                    (None,) + tuple(self.input_shape[1:])
                    if j == self.gemm_input_index
                    else self.model.meta_info[input_name].shape,
                )
                for j, input_name in enumerate(self.node.input)
            ]

            if len(self.node.output) != 1:
                raise ValueError(
                    "Only operations with single outputs can be GEMM-wrapped."
                )

            graph_output = helper.make_tensor_value_info(
                self.node.output[0],
                TensorProto.FLOAT,
                None,  # output dimension is unknown at this point
            )

            graph = helper.make_graph(
                [self.node], "helper-model", graph_inputs, [graph_output]
            )

            model = helper.make_model(graph)
            model.opset_import[0].version = OPSET_VERSION

            # At this point, the model can be checked using:
            # onnx.checker.check_model(model)

            buffer = io.BytesIO()
            onnx.save(model, buffer)
            self.node_inference_session = onnxruntime.InferenceSession(
                buffer.getvalue()
            )

        def run_node_inference(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
            return self.node_inference_session.run(None, inputs)[0]

        def init_output_shape(self) -> None:
            inputs = {
                input: np.zeros(
                    self.meta_info_inputs[j].shape,
                    dtype=_onnx_type_to_numpy(self.meta_info_inputs[j].dtype),
                )
                for j, input in enumerate(self.inputs)
            }

            self.output_shape = list(self.run_node_inference(inputs).shape)

        def init_multiplication_depth(self) -> None:
            self.multiplication_depth = (
                self.meta_info_inputs[self.gemm_input_index].multiplication_depth + 1
            )

        def init_meta_info(self) -> None:
            # note: as only the input with index self.gemm_input_index may be
            # encrypted, the multiplication depth only depends on this input.
            self.init_output_shape()
            self.init_multiplication_depth()

            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=self.multiplication_depth,
                shape=self.output_shape,
                dtype=self.meta_info_inputs[self.gemm_input_index].dtype,
                can_be_encrypted=self.meta_info_inputs[
                    self.gemm_input_index
                ].can_be_encrypted,
            )

        def get_gemm_weights_and_bias(
            self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]
        ) -> Tuple[np.ndarray, np.ndarray]:
            gemm_input_name = self.inputs[self.gemm_input_index]
            common_inputs = {
                input_name: state[input_name]
                for j, input_name in enumerate(self.inputs)
                if j != self.gemm_input_index
            }

            inputs = common_inputs | {
                gemm_input_name: np.zeros(self.input_shape, dtype=np.float32)
            }
            bias = self.run_node_inference(inputs)

            d = np.prod(self.input_shape[1:])
            identity = np.identity(d, dtype=np.float32).reshape(
                [d] + self.input_shape[1:]
            )

            inputs = common_inputs | {gemm_input_name: identity}
            weights = (self.run_node_inference(inputs) - bias).reshape(d, -1)

            return weights, bias.ravel()

        def assert_inputs(
            self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]
        ) -> None:
            if any(
                not isinstance(state[input_name], np.ndarray)
                for j, input_name in enumerate(self.inputs)
                if j != self.gemm_input_index
            ):
                raise ValueError(
                    f"Only the input with index {self.gemm_input_index}"
                    "may be encrypted."
                )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            self.assert_inputs(state)

            x = state[self.inputs[self.gemm_input_index]]
            if isinstance(x, np.ndarray):
                inputs = {input: state[input] for input in self.inputs}
                state[self.output] = self.run_node_inference(inputs)
            else:
                weights, bias = self.get_gemm_weights_and_bias(state)
                result = x @ weights

                if np.any(bias != 0):
                    result += bias

                state[self.output] = result

    class AddOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=max(
                    i.multiplication_depth for i in self.meta_info_inputs
                ),
                shape=list(
                    np.broadcast_shapes(*(i.shape for i in self.meta_info_inputs))
                ),
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=any(
                    [
                        self.meta_info_inputs[i].can_be_encrypted
                        for i in range(len(self.inputs))
                    ]
                ),
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            a = _unwrap_scalar(state[self.inputs[0]])
            b = _unwrap_scalar(state[self.inputs[1]])
            state[self.output] = a + b

    class AveragePoolOperator(GemmWrappedOperator):
        pass

    class ConstantOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            if (attribute := self.attributes.get("value")) is not None:
                self.const = numpy_helper.to_array(attribute.t)
                dtype = attribute.t.data_type
            elif (attribute := self.attributes.get("value_float")) is not None:
                self.const = np.asarray(attribute.f, np.float32)
                dtype = attribute.type
            elif (attribute := self.attributes.get("value_floats")) is not None:
                self.const = np.asarray(attribute.floats, np.float32)
                dtype = attribute.type
            elif (attribute := self.attributes.get("value_int")) is not None:
                self.const = np.asarray(attribute.int, np.int64)
                dtype = attribute.type
            elif (attribute := self.attributes.get("value_ints")) is not None:
                self.const = np.asarray(attribute.ints, np.int64)
                dtype = attribute.type
            else:
                raise NotImplementedError(
                    f"attribute name '{list(self.attributes.keys())[0]}' "
                    "not implemented"
                )

            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=0,
                shape=list(self.const.shape),
                dtype=dtype,
                can_be_encrypted=False,
            )

            self.model._const_state[self.output] = self.const

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            state[self.output] = self.const

    class ConvOperator(GemmWrappedOperator):
        pass

    class FlattenOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)

            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=self.meta_info_inputs[0].multiplication_depth,
                shape=self.meta_info_inputs[0].shape,
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=self.meta_info_inputs[0].can_be_encrypted,
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            x = state[self.inputs[0]]
            if isinstance(x, np.ndarray):
                axis = a.i if (a := self.attributes.get("axis")) is not None else 1
                new_shape = (
                    (1, -1) if axis == 0 else (np.prod(x.shape[:axis]).astype(int), -1)
                )
                x = x.reshape(new_shape)

            state[self.output] = x

    class GemmOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)

            self.trans_a = (a := self.attributes.get("transA")) is not None and a.i != 0
            self.trans_b = (a := self.attributes.get("transB")) is not None and a.i != 0

            shape = [
                self.meta_info_inputs[0].shape[1 if self.trans_a else 0],
                self.meta_info_inputs[1].shape[0 if self.trans_b else 1],
            ]
            multiplication_depth = (
                max(
                    self.meta_info_inputs[0].multiplication_depth,
                    self.meta_info_inputs[1].multiplication_depth,
                )
                + 1
            )

            if len(self.inputs) == 3:
                multiplication_depth = max(
                    multiplication_depth, self.meta_info_inputs[2].multiplication_depth
                )

            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=multiplication_depth,
                shape=shape,
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=any(
                    [
                        self.meta_info_inputs[i].can_be_encrypted
                        for i in range(len(self.inputs))
                    ]
                ),
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            a = state[self.inputs[0]]
            b = state[self.inputs[1]]

            if self.trans_a:
                if isinstance(a, ts.CKKSVector):
                    raise NotImplementedError(
                        "Transpose not implemented for CKKSVector"
                    )
                a = a.T

            if self.trans_b:
                if isinstance(b, ts.CKKSVector):
                    raise NotImplementedError(
                        "Transpose not implemented for CKKSVector"
                    )
                b = b.T

            if (attribute := self.attributes.get("alpha")) is not None:
                b = b * attribute.f

            result = a @ b

            if len(self.inputs) == 3:
                c = state[self.inputs[2]]
                if (attribute := self.attributes.get("beta")) is not None:
                    c = c * attribute.f
                result = result + c

            state[self.output] = result

    class MatMulOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            multiplication_depth = 1 + max(
                i.multiplication_depth for i in self.meta_info_inputs
            )
            shape = [
                self.meta_info_inputs[0].shape[-2],
                self.meta_info_inputs[1].shape[-1],
            ]
            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=multiplication_depth,
                shape=shape,
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=any(
                    [
                        self.meta_info_inputs[i].can_be_encrypted
                        for i in range(len(self.inputs))
                    ]
                ),
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            state[self.output] = state[self.inputs[0]] @ state[self.inputs[1]]

    class MulOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=(
                    max(i.multiplication_depth for i in self.meta_info_inputs) + 1
                ),
                shape=list(
                    np.broadcast_shapes(*(i.shape for i in self.meta_info_inputs))
                ),
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=any(
                    [
                        self.meta_info_inputs[i].can_be_encrypted
                        for i in range(len(self.inputs))
                    ]
                ),
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            a = _unwrap_scalar(state[self.inputs[0]])
            b = _unwrap_scalar(state[self.inputs[1]])
            state[self.output] = a * b

    # PadOperator would waste a multiplication if we just derive it from GemmWrapped
    # without overwriting init_multiplication_depth and execute
    class PadOperator(GemmWrappedOperator):
        def init_output_shape(self) -> None:
            pads = self.model._const_state.get(self.inputs[1])
            if pads is None:
                raise NotImplementedError("Only const pads are supported")

            self.pads = pads

            # cast to int, as pads are np.int64 and shape requires int or str
            self.output_shape = [
                int(d + self.pads[j] + self.pads[len(self.input_shape) + j])
                for j, d in enumerate(self.input_shape)
            ]

        def init_multiplication_depth(self) -> None:
            self.multiplication_depth = (
                self.meta_info_inputs[self.gemm_input_index].multiplication_depth + 0
                if all(pad == 0 for pad in self.pads)
                else 1
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            if all(pad == 0 for pad in self.pads):
                state[self.output] = state[self.inputs[0]]
            else:
                super().execute(state)

    class ReluOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            self.degree = 0
            self.no_offset = False
            relu_mode = self.model.relu_mode

            if relu_mode is None:
                raise ValueError("Missing relu_mode.")
            elif relu_mode.startswith("deg"):
                self.degree = int(relu_mode.split("_")[0][3:])
                if self.degree < 1:
                    raise ValueError("Invalid degree for relu approximation.")
                if relu_mode.endswith("no_offset"):
                    self.no_offset = True
            else:
                raise ValueError("Invalid ReLU approximation mode.")

            self.relu_mode = relu_mode

            # plus one for the coefficient multiplication
            relu_multiplications = floor(log2(self.degree)) + 1

            multiplication_depth = relu_multiplications + max(
                i.multiplication_depth for i in self.meta_info_inputs
            )

            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=multiplication_depth,
                shape=self.meta_info_inputs[0].shape,
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=self.meta_info_inputs[0].can_be_encrypted,
            )

        def _relu(self, x: np.ndarray) -> np.ndarray:
            return np.max([x, np.zeros_like(x)], axis=0)

        def _compute_polynomial_coeffs(self, interval: Interval) -> List[float]:
            lower = interval.lower_bound * self.model._domain_factors[0]
            upper = interval.upper_bound * self.model._domain_factors[1]

            if lower >= 0:
                # return linear model
                return [1, 0]
            elif upper <= 0:
                # this would give a zero model
                raise ValueError("Relu would produce a zero model.")

            x_eval = np.linspace(lower, upper, 1000)
            coeffs = np.polyfit(x_eval, self._relu(x_eval), self.degree).tolist()
            if self.no_offset:
                coeffs[-1] = 0

            return coeffs

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            x = state[self.inputs[0]]

            if self.meta_info_inputs[0].domain is not None:
                domain = self.meta_info_inputs[0].domain
            elif isinstance(x, np.ndarray):
                # calibration case
                domain = Interval(np.amin(x), np.amax(x))
            else:
                raise ValueError(
                    "ReLU requires a calibrated model for encrypted inference."
                )

            coeffs = self._compute_polynomial_coeffs(domain)

            if isinstance(x, np.ndarray):
                y = np.polyval(coeffs, x)
            else:
                y = x.polyval(coeffs[::-1])

            state[self.output] = y

    class ReshapeOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)

            target_shape = self.model._const_state.get(self.inputs[1])
            if target_shape is None:
                raise NotImplementedError(
                    "ReshapeOperator is only implemented for constant output shape"
                )
            self.target_shape = [int(d) for d in target_shape]

            minus_one_indices = [j for j, d in enumerate(self.target_shape) if d == -1]
            if len(minus_one_indices) > 1:
                raise ValueError(
                    "Only one reshape dimension can be -1, but reshape dimensions "
                    f"are {self.target_shape}"
                )

            input_shape = self.meta_info_inputs[0].shape
            input_size = np.prod(input_shape)

            output_meta_info_shape = list(self.target_shape)
            if len(minus_one_indices) == 1:
                output_meta_info_shape[minus_one_indices[0]] = -int(
                    input_size // np.prod(self.target_shape)
                )

            if np.prod(output_meta_info_shape) != input_size:
                raise ValueError(
                    f"Input of shape {input_shape} cannot be reshaped to shape "
                    f"{output_meta_info_shape}"
                )

            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=self.meta_info_inputs[0].multiplication_depth,
                shape=output_meta_info_shape,
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=self.meta_info_inputs[0].can_be_encrypted,
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            x = state[self.inputs[0]]
            if isinstance(x, ts.CKKSVector):
                state[self.output] = x
            else:
                state[self.output] = x.reshape(self.target_shape)

    class SubOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=(
                    max(i.multiplication_depth for i in self.meta_info_inputs)
                ),
                shape=list(
                    np.broadcast_shapes(*(i.shape for i in self.meta_info_inputs))
                ),
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=True,
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            a = _unwrap_scalar(state[self.inputs[0]])
            b = _unwrap_scalar(state[self.inputs[1]])
            state[self.output] = a - b

    # START: CLEARTEXT ONLY ###

    class DivOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=(
                    max(i.multiplication_depth for i in self.meta_info_inputs)
                ),
                shape=list(
                    np.broadcast_shapes(*(i.shape for i in self.meta_info_inputs))
                ),
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=True,
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            a = _unwrap_scalar(state[self.inputs[0]])
            b = _unwrap_scalar(state[self.inputs[1]])
            if any(isinstance(x, ts.CKKSVector) for x in (a, b)):
                raise NotImplementedError("DivOperator not implemented for CKKSVector")

            state[self.output] = a / b

    class BatchNormalizationOperator(Operator):
        def __init__(self, model: "ONNXModel", node: onnx.onnx_ml_pb2.NodeProto):
            super().__init__(model, node)
            self.model.meta_info[self.output] = TensorMetaInfo(
                multiplication_depth=(
                    max(i.multiplication_depth for i in self.meta_info_inputs) + 1
                ),
                shape=self.meta_info_inputs[0].shape,
                dtype=self.meta_info_inputs[0].dtype,
                can_be_encrypted=True,
            )

        def execute(self, state: Dict[str, Union[ts.CKKSVector, np.ndarray]]) -> None:
            X = state[self.inputs[0]]
            if isinstance(X, ts.CKKSVector):
                raise NotImplementedError(
                    "BatchNormalizationOperator not implemented for CKKSVector"
                )

            scale = state[self.inputs[1]]
            B = state[self.inputs[2]]
            input_mean = state[self.inputs[3]]
            input_var = state[self.inputs[4]]
            epsilon = a.f if (a := self.attributes.get("epsilon")) is not None else 1e-5

            dims_x = len(X.shape)
            dim_ones = (1,) * (dims_x - 2)
            scale = scale.reshape(-1, *dim_ones)
            B = B.reshape(-1, *dim_ones)
            input_mean = input_mean.reshape(-1, *dim_ones)
            input_var = input_var.reshape(-1, *dim_ones)

            Y = (X - input_mean) / np.sqrt(input_var + epsilon) * scale + B

            state[self.output] = Y


def _unwrap_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return float(x)
    else:
        return x


def _onnx_type_to_numpy(dtype: int) -> np.dtype:
    # note: UNDEFINED and BFLOAT16 are not supported
    result = {
        TensorProto.DataType.FLOAT: np.float32,
        TensorProto.DataType.UINT8: np.uint8,
        TensorProto.DataType.INT8: np.int8,
        TensorProto.DataType.UINT16: np.uint16,
        TensorProto.DataType.INT16: np.int16,
        TensorProto.DataType.INT32: np.int32,
        TensorProto.DataType.INT64: np.int64,
        TensorProto.DataType.STRING: np.str_,
        TensorProto.DataType.BOOL: np.bool8,
        TensorProto.DataType.FLOAT16: np.float16,
        TensorProto.DataType.DOUBLE: np.double,
        TensorProto.DataType.UINT32: np.uint32,
        TensorProto.DataType.UINT64: np.uint64,
        TensorProto.DataType.COMPLEX64: np.complex64,
        TensorProto.DataType.COMPLEX128: np.complex128,
    }.get(dtype)

    if result is not None:
        return result
    else:
        dtype_name = next(
            (name for name, j in TensorProto.DataType.items() if j == dtype), None
        )
        if dtype_name is not None:
            raise ValueError(f"Type '{dtype_name}' not supported")
        else:
            raise ValueError(f"Unknown type index {dtype}")
