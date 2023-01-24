import pytest
from test_definitions import (
    APPROXIMATED_MODELS_DIR,
    CALIBRATION_DATA_DIR,
    FIXED_DEPTH_MODELS_DIR,
    MODEL_DIR,
)

from tenseal_inference.config import KeyParamsConfig
from tenseal_inference.crypto import find_optimal_parameters
from tenseal_inference.inference import ONNXModel


@pytest.mark.parametrize(
    "keyparams_cfg,n_bits_int_precision_expected,parameters_expected",
    [
        (
            # required [32, 25, 25, 25, 32] (sum 139) can be improved
            KeyParamsConfig(
                model_path=FIXED_DEPTH_MODELS_DIR / "depth3.onnx",
                key_params_path="/",
                n_bits_fractional_precision=25,
                calibration_data_path=CALIBRATION_DATA_DIR
                / "lower_0_upper_100.npz",  # ==> n_bits_int_precision=7
                relu_mode="deg3",
                domain_mode="min-max",
            ),
            7,
            (8192, [49, 40, 40, 40, 49]),
        ),
        (
            # requried [40, 29, 40] (sum 109) cannot be improved
            KeyParamsConfig(
                model_path=FIXED_DEPTH_MODELS_DIR / "depth1.onnx",
                key_params_path="/",
                n_bits_fractional_precision=29,
                calibration_data_path=CALIBRATION_DATA_DIR
                / "lower_2000_upper_2000.npz",  # ==> n_bits_int_precision=11
                relu_mode="deg3",
                domain_mode="min-max",
            ),
            11,
            (4096, [40, 29, 40]),
        ),
        (
            # requried [40, 30, 40] (sum 110) must not be improved beyond [60, 50, 60]
            KeyParamsConfig(
                model_path=FIXED_DEPTH_MODELS_DIR / "depth1.onnx",
                key_params_path="/",
                n_bits_fractional_precision=30,
                calibration_data_path=CALIBRATION_DATA_DIR
                / "lower_-512_upper_0.npz",  # ==> n_bits_int_precision=10
                relu_mode="deg3",
                domain_mode="min-max",
            ),
            10,
            (8192, [60, 50, 60]),
        ),
        (
            # required [25, 22, 22, 25] (sum 96) can be improved to [29, 25, 25, 29]
            KeyParamsConfig(
                model_path=APPROXIMATED_MODELS_DIR / "relu.onnx",
                key_params_path="/",
                n_bits_fractional_precision=22,
                calibration_data_path=CALIBRATION_DATA_DIR
                / "lower_0_upper_5.npz",  # ==> n_bits_int_pre
                relu_mode="deg3",
                domain_mode="min-max",
            ),
            3,
            (4096, [29, 25, 25, 29]),
        ),
    ],
)
def test_find_optimal_parameters(
    keyparams_cfg, n_bits_int_precision_expected, parameters_expected
):
    model = ONNXModel(keyparams_cfg.model_path, keyparams_cfg)
    assert model.n_bits_integer_precision == n_bits_int_precision_expected
    key_params = find_optimal_parameters(keyparams_cfg, model)
    assert key_params.poly_modulus_degree == parameters_expected[0]
    assert key_params.coeff_mod_bit_sizes == parameters_expected[1]


@pytest.mark.parametrize(
    "keyparams_cfg",
    [
        (
            KeyParamsConfig(
                model_path=MODEL_DIR / "power2-plus-power4.onnx",
                key_params_path="/",
                n_bits_fractional_precision=30,
                calibration_data_path=CALIBRATION_DATA_DIR / "lower_0_upper_200.npz",
                relu_mode="deg3",
                domain_mode="min-max",
            )  # 31 bits integer + 30 bits fractional precision > 60 bits
        )
    ],
)
def test_model_resulting_in_invalid_keyparams(keyparams_cfg):
    model = ONNXModel(keyparams_cfg.model_path, keyparams_cfg)
    with pytest.raises(ValueError):
        find_optimal_parameters(keyparams_cfg, model)
