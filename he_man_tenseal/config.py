import re
from pathlib import Path
from typing import Callable, Optional

import click
from pydantic_settings import BaseSettings


class GlobalSettings(BaseSettings):
    pass


class ModelInput(BaseSettings):
    _model_path: Callable = click.option(
        "-m",
        "--model-path",
        type=click.Path(readable=True),
        required=True,
        help="Path of the model to be used",
    )
    model_path: Path


class PrecisionInput(BaseSettings):
    _n_bits_fractional_precision: Callable = click.option(
        "-f",
        "--n-bits-fractional-precision",
        type=click.IntRange(21),
        default=21,
        help=(
            "Lower bound for number of bits to represent fractional part of "
            "ciphertexts resulting from the key to be generated"
        ),
    )
    n_bits_fractional_precision: int


class KeyParamsOutput(BaseSettings):
    _key_params_path: Callable = click.option(
        "-o",
        "--key-params-path",
        type=click.Path(writable=True),
        required=True,
        help="Path to store the key parameters",
    )
    key_params_path: Path


class KeyParamsInput(BaseSettings):
    _key_params_path: Callable = click.option(
        "-i",
        "--key-params-path",
        type=click.Path(readable=True),
        required=True,
        help="Path to load the key parameters from",
    )
    key_params_path: Path


class KeyOutput(BaseSettings):
    _secret_key_path: Callable = click.option(
        "-o",
        "--secret-key-path",
        type=click.Path(writable=True),
        required=True,
        help="Path to store the generated secret key",
    )
    secret_key_path: Path


class KeyInput(BaseSettings):
    _key_path: Callable = click.option(
        "-k",
        "--key-path",
        type=click.Path(readable=True),
        required=True,
        help="Path to load the key from",
    )
    key_path: Path


class PlaintextOutput(BaseSettings):
    _plaintext_output_path: Callable = click.option(
        "-o",
        "--plaintext-output-path",
        type=click.Path(writable=True),
        required=True,
        help="Path of the file to store the plaintext output",
    )
    plaintext_output_path: Path


class PlaintextInput(BaseSettings):
    _plaintext_input_path: Callable = click.option(
        "-i",
        "--plaintext-input-path",
        type=click.Path(readable=True),
        required=True,
        help="Path of the file containing the plaintext input",
    )
    plaintext_input_path: Path


class CiphertextOutput(BaseSettings):
    _ciphertext_output_path: Callable = click.option(
        "-o",
        "--ciphertext-output-path",
        type=click.Path(writable=True),
        required=True,
        help="Path of the file to store the ciphertext output",
    )
    ciphertext_output_path: Path


class CiphertextInput(BaseSettings):
    _ciphertext_input_path: Callable = click.option(
        "-i",
        "--ciphertext-input-path",
        type=click.Path(readable=True),
        required=True,
        help="Path of the file containing the ciphertext input",
    )
    ciphertext_input_path: Path


class CalibrationDataInput(BaseSettings):
    _calibration_data_path: Callable = click.option(
        "-c",
        "--calibration-data-path",
        type=click.Path(readable=True),
        required=True,
        help="Path of the zip file containing the calibration data",
    )
    calibration_data_path: Path


def check_relu_mode(
    ctx: Optional[click.Context], param: Optional[click.Parameter], value: str
) -> str:
    value = value.lower()
    if re.match("^deg[1-9][0-9]*(_no_offset)?$", value):
        return value
    else:
        raise click.BadParameter("Invalid relu_mode parameter.")


class ReluApproximationMode(BaseSettings):
    _relu_mode: Callable = click.option(
        "--relu-mode",
        type=click.STRING,
        callback=check_relu_mode,
        default="deg3",
        required=False,
        help="Method to approximate ReLU using a polynomial",
    )
    relu_mode: str


class DomainCalibrationMode(BaseSettings):
    _domain_mode: Callable = click.option(
        "--domain-mode",
        type=click.Choice(["min-max", "mean-std"]),
        default="min-max",
        required=False,
        help="Method to calibrate the domain",
    )
    domain_mode: str


class KeyParamsConfig(
    GlobalSettings,
    ModelInput,
    PrecisionInput,
    CalibrationDataInput,
    ReluApproximationMode,
    DomainCalibrationMode,
    KeyParamsOutput,
):
    pass


class KeyGenConfig(GlobalSettings, KeyParamsInput, KeyOutput):
    pass


class EncryptConfig(GlobalSettings, KeyInput, PlaintextInput, CiphertextOutput):
    pass


class InferenceConfig(
    GlobalSettings, ModelInput, KeyInput, CiphertextInput, CiphertextOutput
):
    pass


class DecryptConfig(GlobalSettings, KeyInput, CiphertextInput, PlaintextOutput):
    pass
