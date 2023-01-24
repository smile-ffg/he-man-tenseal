import os
from pathlib import Path

import numpy as np

from tenseal_inference.config import (
    DecryptConfig,
    EncryptConfig,
    KeyGenConfig,
    KeyParamsConfig,
)
from tenseal_inference.main import run_decrypt, run_encrypt, run_keygen, run_keyparams


def test_encrypt_decrypt(tmp_path):
    model_dir = Path(__file__).parent / "models-fixed-depth"
    model_path = model_dir / "depth1.onnx"
    calibration_data_dir = Path(__file__).parent / "calibration-data"
    key_params_path = tmp_path / "keyparams.json"
    key_path = tmp_path / "key"
    plaintext_input_path = tmp_path / "plaintext.npy"
    ciphertext_path = tmp_path / "ciphertext.enc"
    plaintext_output_path = tmp_path / "plaintext2.npy"

    run_keyparams(
        KeyParamsConfig(
            key_params_path=key_params_path,
            model_path=model_path,
            n_bits_fractional_precision=30,
            calibration_data_path=calibration_data_dir
            / "lower_0_upper_1000.npz",  # ==> n_bits_int_precision=10
            relu_mode="deg3",
            domain_mode="min-max",
        )
    )

    run_keygen(
        KeyGenConfig(
            key_params_path=key_params_path,
            secret_key_path=key_path,
        )
    )

    plaintext = np.arange(16.0)
    np.save(plaintext_input_path, plaintext)

    run_encrypt(
        EncryptConfig(
            key_path=key_path,
            plaintext_input_path=plaintext_input_path,
            ciphertext_output_path=ciphertext_path,
        )
    )

    run_decrypt(
        DecryptConfig(
            key_path=key_path,
            ciphertext_input_path=ciphertext_path,
            plaintext_output_path=plaintext_output_path,
        )
    )

    plaintext2 = np.load(plaintext_output_path)

    assert np.allclose(plaintext, plaintext2, atol=1e-3)

    # delete generated calibrated-model
    calibrated_model_path = model_path.parent / (
        model_path.stem + "_calibrated" + model_path.suffix
    )
    if calibrated_model_path.exists():
        os.remove(calibrated_model_path)


def test_encrypt_with_evaluation_key(tmp_path):
    model_dir = Path(__file__).parent / "models-fixed-depth"
    model_path = model_dir / "depth1.onnx"
    calibration_data_dir = Path(__file__).parent / "calibration-data"
    key_params_path = tmp_path / "keyparams.json"
    key_path = tmp_path / "key"
    plaintext_input_path = tmp_path / "plaintext.npy"
    ciphertext_path = tmp_path / "ciphertext.enc"
    plaintext_output_path = tmp_path / "plaintext2.npy"

    run_keyparams(
        KeyParamsConfig(
            key_params_path=key_params_path,
            model_path=model_path,
            n_bits_fractional_precision=30,
            calibration_data_path=calibration_data_dir
            / "lower_0_upper_1000.npz",  # ==> n_bits_int_precision=10
            relu_mode="deg3",
            domain_mode="min-max",
        )
    )

    run_keygen(
        KeyGenConfig(
            key_params_path=key_params_path,
            secret_key_path=key_path,
        )
    )

    plaintext = np.arange(16.0)
    np.save(plaintext_input_path, plaintext)

    run_encrypt(
        EncryptConfig(
            key_path=Path(f"{key_path}.pub"),
            plaintext_input_path=plaintext_input_path,
            ciphertext_output_path=ciphertext_path,
        )
    )

    run_decrypt(
        DecryptConfig(
            key_path=key_path,
            ciphertext_input_path=ciphertext_path,
            plaintext_output_path=plaintext_output_path,
        )
    )

    plaintext2 = np.load(plaintext_output_path)

    assert np.allclose(plaintext, plaintext2, atol=1e-3)

    # delete generated calibrated-model
    calibrated_model_path = model_path.parent / (
        model_path.stem + "_calibrated" + model_path.suffix
    )
    if calibrated_model_path.exists():
        os.remove(calibrated_model_path)
