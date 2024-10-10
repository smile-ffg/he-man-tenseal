import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import onnx
import onnxruntime
import pytest
from loguru import logger
from he_man_tenseal.util import load_calibration_data
from test_definitions import (
    APPROXIMATED_MODELS_DIR,
    CALIBRATION_DATA_DIR,
    INPUTS_DIR,
    MODEL_DIR,
)

from he_man_tenseal.config import (
    DecryptConfig,
    EncryptConfig,
    InferenceConfig,
    KeyGenConfig,
    KeyParamsConfig,
)
from he_man_tenseal.inference import ONNXModel
from he_man_tenseal.main import (
    run_decrypt,
    run_encrypt,
    run_inference,
    run_keygen,
    run_keyparams,
)


def test_inference(tmp_path):
    try:
        # generate keys
        secret_key_path = tmp_path / "key"
        evaluation_key_path = Path(f"{secret_key_path}.pub")

        # note: a depth 2 key is sufficient for all test models
        key_params_path = tmp_path / "keyparams.json"
        run_keyparams(
            KeyParamsConfig(
                key_params_path=key_params_path,
                onnx_path=MODEL_DIR / "power2-plus-power4.onnx",
                n_bits_fractional_precision=50,
                calibration_data_path=CALIBRATION_DATA_DIR
                / "lower_0_upper_5.npz",  # ==> n_bits_int_precision = 10
                relu_mode="deg3",
                domain_mode="min-max",
                split=0,
            )
        )

        run_keygen(
            KeyGenConfig(
                key_params_path=key_params_path,
                secret_key_path=secret_key_path,
            )
        )

        for model_filename in os.listdir(MODEL_DIR):
            model_path = MODEL_DIR / model_filename
            model = onnx.load(model_path)

            # create input
            np.random.seed(42)
            x = np.asarray(
                np.random.random(
                    [
                        d.dim_value
                        for d in model.graph.input[0].type.tensor_type.shape.dim
                    ]
                ),
                np.float32,
            )
            plaintext_input_path = tmp_path / "input.npy"
            np.save(plaintext_input_path, x[0])

            # evaluate plaintext model
            session = onnxruntime.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            y = session.run(None, {model.graph.input[0].name: x})[0]

            # encrypt input
            ciphertext_input_path = tmp_path / "input.enc"
            run_encrypt(
                EncryptConfig(
                    key_path=secret_key_path,
                    plaintext_input_path=plaintext_input_path,
                    ciphertext_output_path=ciphertext_input_path,
                )
            )

            # run encrypted inference
            ciphertext_output_path = tmp_path / "output.enc"
            run_inference(
                InferenceConfig(
                    onnx_path=model_path,
                    key_path=evaluation_key_path,
                    input_path=[ciphertext_input_path],
                    output_path=[ciphertext_output_path],
                )
            )

            # decrypt output
            plaintext_output_path = tmp_path / "output.npy"
            run_decrypt(
                DecryptConfig(
                    key_path=secret_key_path,
                    ciphertext_input_path=ciphertext_output_path,
                    plaintext_output_path=plaintext_output_path,
                )
            )

            # check result
            y2 = np.load(plaintext_output_path)
            logger.debug(
                f"max. abs error: {abs(y - y2.reshape(y.shape)).max()}"
                f" @ {model_filename}"
            )
            assert np.allclose(y, y2.reshape(y.shape), atol=2e-7, rtol=1e-4)

    finally:
        # delete calibrated model
        calibrated_model_path = MODEL_DIR / "power2-plus-power4_calibrated.onnx"
        if calibrated_model_path.exists():
            os.remove(calibrated_model_path)


def test_onnx_model_call_with_wrong_number_of_inputs():
    model_path = MODEL_DIR / "matmul.onnx"
    model = ONNXModel(model_path)
    with pytest.raises(ValueError):
        model(*([None] * (model.n_inputs + 1)))


@pytest.mark.parametrize(
    "model_filename, multiplication_depth",
    [
        ("add.onnx", 0),
        ("gemm.onnx", 1),
        ("matmul.onnx", 1),
        ("mul.onnx", 1),
        ("power2-plus-power4.onnx", 2),
        ("conv.onnx", 1),
    ],
)
def test_multiplication_depth(model_filename, multiplication_depth):
    model = ONNXModel(path=MODEL_DIR / model_filename)
    assert model.multiplication_depth == multiplication_depth


@pytest.mark.parametrize(
    "keyparams_cfg, n_bits_integer_precision",
    [
        (
            KeyParamsConfig(
                onnx_path=MODEL_DIR / "3-bit-int-model.onnx",
                key_params_path="/",
                n_bits_fractional_precision=50,
                calibration_data_path=CALIBRATION_DATA_DIR / "lower_-1_upper_1.npz",
                relu_mode="deg3",
                domain_mode="min-max",
                split=0,
            ),
            3,
        )
    ],
)
def test_int_precision(keyparams_cfg, n_bits_integer_precision):
    model = ONNXModel(keyparams_cfg.onnx_path, keyparams_cfg)
    calibration_data = load_calibration_data(keyparams_cfg.calibration_data_path)
    model.calibrate(calibration_data)
    assert model.n_bits_integer_precision == n_bits_integer_precision


def test_mnist_relu_inference(tmp_path):
    # generate keys
    secret_key_path = tmp_path / "key"
    evaluation_key_path = Path(f"{secret_key_path}.pub")

    model_path = APPROXIMATED_MODELS_DIR / "mnist_small_relu.onnx"
    calibrated_model_path = model_path.parent / (
        model_path.stem + "_calibrated" + model_path.suffix
    )

    key_params_path = tmp_path / "keyparams.json"
    run_keyparams(
        KeyParamsConfig(
            key_params_path=key_params_path,
            onnx_path=model_path,
            n_bits_fractional_precision=22,
            calibration_data_path=CALIBRATION_DATA_DIR / "mnist_28x28.zip",
            relu_mode="deg3",
            domain_mode="min-max",
            split=0,
        )
    )

    run_keygen(
        KeyGenConfig(
            key_params_path=key_params_path,
            secret_key_path=secret_key_path,
        )
    )

    model = onnx.load(calibrated_model_path)

    plaintext_input_path = INPUTS_DIR / "mnist_28x28_7.npy"
    x = np.load(plaintext_input_path)

    # evaluate plaintext model
    session = onnxruntime.InferenceSession(
        str(calibrated_model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    y = session.run(None, {model.graph.input[0].name: x})[0]

    # encrypt input
    ciphertext_input_path = tmp_path / "input.enc"
    run_encrypt(
        EncryptConfig(
            key_path=secret_key_path,
            plaintext_input_path=plaintext_input_path,
            ciphertext_output_path=ciphertext_input_path,
        )
    )

    # run encrypted inference
    ciphertext_output_path = tmp_path / "output.enc"
    run_inference(
        InferenceConfig(
            onnx_path=calibrated_model_path,
            key_path=evaluation_key_path,
            input_path=[ciphertext_input_path],
            output_path=[ciphertext_output_path],
        )
    )

    # decrypt output
    plaintext_output_path = tmp_path / "output.npy"
    run_decrypt(
        DecryptConfig(
            key_path=secret_key_path,
            ciphertext_input_path=ciphertext_output_path,
            plaintext_output_path=plaintext_output_path,
        )
    )

    # check result
    y2 = np.load(plaintext_output_path)
    assert np.argmax(y) == np.argmax(y2)

    # delete calibrated model
    if calibrated_model_path.exists():
        os.remove(calibrated_model_path)


def test_inference_split(tmp_path):
    model_path = tmp_path / "model.onnx"
    copyfile(MODEL_DIR / "power2-plus-power4.onnx", model_path)
    preprocessing_model_path = tmp_path / "model_preprocessing.onnx"
    calibrated_core_model_path = tmp_path / "model_core_calibrated.onnx"
    calibration_data_path = CALIBRATION_DATA_DIR / "lower_-1_upper_1.npz"
    key_params_path = tmp_path / "keyparams.json"
    secret_key_path = tmp_path / "key"
    evaluation_key_path = Path(f"{secret_key_path}.pub")
    plaintext_input = np.asarray([-1, -0.42, 0, 0.84, 1], dtype=np.float32)
    plaintext_input_path = tmp_path / "input.npy"
    np.save(plaintext_input_path, plaintext_input)
    intermediate_plaintext_path = tmp_path / "intermediate.npy"
    intermediate_ciphertext_path = tmp_path / "intermediate.enc"
    ciphertext_output_path = tmp_path / "output.enc"
    plaintext_output_path = tmp_path / "output.npy"
    test_output_path = tmp_path / "output_test.npy"

    run_inference(
        InferenceConfig(
            onnx_path=model_path,
            key_path=None,
            input_path=[plaintext_input_path],
            output_path=[test_output_path],
        )
    )
    assert test_output_path.is_file()
    test_output = np.load(test_output_path)

    run_keyparams(
        KeyParamsConfig(
            key_params_path=key_params_path,
            onnx_path=model_path,
            n_bits_fractional_precision=50,
            calibration_data_path=calibration_data_path,
            relu_mode="deg3",
            domain_mode="min-max",
            split=1,
        )
    )
    assert key_params_path.is_file()
    assert preprocessing_model_path.is_file()
    assert calibrated_core_model_path.is_file()

    run_keygen(
        KeyGenConfig(
            key_params_path=key_params_path,
            secret_key_path=secret_key_path,
        )
    )
    assert secret_key_path.is_file()
    assert evaluation_key_path.is_file()

    run_inference(
        InferenceConfig(
            onnx_path=preprocessing_model_path,
            key_path=None,
            input_path=[plaintext_input_path],
            output_path=[intermediate_plaintext_path],
        )
    )
    assert intermediate_plaintext_path.is_file()

    run_encrypt(
        EncryptConfig(
            key_path=secret_key_path,
            plaintext_input_path=intermediate_plaintext_path,
            ciphertext_output_path=intermediate_ciphertext_path,
        )
    )
    assert intermediate_ciphertext_path.is_file()

    run_inference(
        InferenceConfig(
            onnx_path=calibrated_core_model_path,
            key_path=evaluation_key_path,
            input_path=[intermediate_ciphertext_path],
            output_path=[ciphertext_output_path],
        )
    )
    assert ciphertext_output_path.is_file()

    run_decrypt(
        DecryptConfig(
            key_path=secret_key_path,
            ciphertext_input_path=ciphertext_output_path,
            plaintext_output_path=plaintext_output_path,
        )
    )
    assert plaintext_output_path.is_file()
    output = np.load(plaintext_output_path)

    assert np.allclose(output, test_output, rtol=1e-5, atol=1e-8)
