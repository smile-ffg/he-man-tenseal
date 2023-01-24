import sys
from typing import Any, Callable

import click
import numpy as np
import tenseal as ts
from loguru import logger
from pydantic import BaseSettings

from he_man_tenseal import config, crypto
from he_man_tenseal.inference import ONNXModel


def config_args(cfg_class: BaseSettings) -> Callable:
    def annotator(function: Callable) -> Callable:
        for key in cfg_class.__fields__:
            if hasattr(cfg_class, "_%s" % key):
                function = getattr(cfg_class, "_%s" % key)(function)
        return function

    return annotator


@click.group()
@click.version_option()
@click.option("-v", "--verbose", count=True, help="set the verbosity, can be repeated")
@logger.catch
def command_line(verbose: int) -> None:
    """A CLI tool for model inference with TenSEAL"""
    logger.remove()
    logger.add(
        sys.stderr,
        level={0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG", 4: "TRACE"}[verbose],
    )


@command_line.command()
@config_args(config.KeyParamsConfig)
@logger.catch
def keyparams(**kwargs: Any) -> None:
    """Generates HE keys"""
    try:
        cfg = config.KeyParamsConfig(**kwargs)
        run_keyparams(cfg)
    except Exception as e:
        raise click.ClickException(str(e))


@command_line.command()
@config_args(config.KeyGenConfig)
@logger.catch
def keygen(**kwargs: Any) -> None:
    """Generates HE keys"""
    try:
        cfg = config.KeyGenConfig(**kwargs)
        run_keygen(cfg)
    except Exception as e:
        raise click.ClickException(str(e))


@command_line.command()
@config_args(config.EncryptConfig)
@logger.catch
def encrypt(**kwargs: Any) -> None:
    """Encrypts model input"""
    try:
        cfg = config.EncryptConfig(**kwargs)
        run_encrypt(cfg)
    except Exception as e:
        raise click.ClickException(str(e))


@command_line.command()
@config_args(config.InferenceConfig)
@logger.catch
def inference(**kwargs: Any) -> None:
    """Applies a model to an encrypted input"""
    try:
        cfg = config.InferenceConfig(**kwargs)
        run_inference(cfg)
    except Exception as e:
        raise click.ClickException(str(e))


@command_line.command()
@config_args(config.DecryptConfig)
@logger.catch
def decrypt(**kwargs: Any) -> None:
    """Decrypts model output"""
    try:
        cfg = config.DecryptConfig(**kwargs)
        run_decrypt(cfg)
    except Exception as e:
        raise click.ClickException(str(e))


def run_keyparams(cfg: config.KeyParamsConfig) -> None:
    model = ONNXModel(cfg.model_path, cfg)
    key_params = crypto.find_optimal_parameters(cfg, model)
    key_params.save(cfg.key_params_path)
    model.save_calibrated_model()


def run_keygen(cfg: config.KeyGenConfig) -> None:
    key_params = crypto.KeyParams.load(cfg.key_params_path)
    context = crypto.create_context(key_params)
    crypto.save_context(context, cfg.secret_key_path)


def run_encrypt(cfg: config.EncryptConfig) -> None:
    context = crypto.load_context(cfg.key_path)
    plaintext = np.load(cfg.plaintext_input_path)
    ciphertext = ts.ckks_vector(context, plaintext.ravel())
    crypto.save_vector(ciphertext, cfg.ciphertext_output_path)


def run_inference(cfg: config.InferenceConfig) -> None:
    model = ONNXModel(cfg.model_path)
    context = crypto.load_context(cfg.key_path)
    input = crypto.load_vector(context, cfg.ciphertext_input_path)
    output = model(input)[0]
    crypto.save_vector(output, cfg.ciphertext_output_path)


def run_decrypt(cfg: config.DecryptConfig) -> None:
    context = crypto.load_context(cfg.key_path)
    ciphertext = crypto.load_vector(context, cfg.ciphertext_input_path)
    plaintext = ciphertext.decrypt()
    np.save(cfg.plaintext_output_path, plaintext)
