import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tenseal as ts
from loguru import logger

from he_man_tenseal.inference import ONNXModel

from .config import KeyParamsConfig

# maximum allowed bit size sums by poly modulus degree (for 128 bit security)
_MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE = {
    1024: 27,
    2048: 54,
    4096: 109,
    8192: 218,
    16384: 438,
    32768: 881,
}


@dataclass
class KeyParams:
    poly_modulus_degree: int
    coeff_mod_bit_sizes: List[int]

    def save(self, path: Path) -> None:
        obj = {"library": "seal", "parameters": dataclasses.asdict(self)}
        with open(path, "w") as output_file:
            json.dump(obj, output_file)

    @staticmethod
    def load(path: Path) -> "KeyParams":
        with open(path, "r") as input_file:
            obj = json.load(input_file)
        return KeyParams(**obj["parameters"])


def create_context(key_params: KeyParams) -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=key_params.poly_modulus_degree,
        coeff_mod_bit_sizes=key_params.coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** key_params.coeff_mod_bit_sizes[-2]
    return context


def find_min_poly_modulus_degree(cfg: KeyParamsConfig, model: ONNXModel) -> int:
    """Finds the minimal possible poly modulus degree for the given keyparameter config.

    Args:
        cfg (KeyParamsConfig): The key parameter config.

    Returns:
        int: Minimal possible poly modulus degree.
    """
    # compute the minimum possible bit size sum
    min_bit_size_sum = (
        2 * model.n_bits_integer_precision
        + (model.multiplication_depth + 2) * cfg.n_bits_fractional_precision
    )
    logger.trace(f"minimum bit size sum is {min_bit_size_sum}")

    if min_bit_size_sum > max(_MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE.values()):
        raise ValueError("minimum bit size exceeds maximum bit size threshold")

    # from this, the poly modulus degree can be derived, which also fixes the
    # maximum bit size sum
    poly_modulus_degree, max_bit_size_sum = min(
        (n, s)
        for n, s in _MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE.items()
        if s >= min_bit_size_sum and n >= 2 * model.max_encrypted_size
    )
    logger.trace(
        f"poly_modulus_degree {poly_modulus_degree} selected "
        f"(max_bit_size_sum = {max_bit_size_sum})"
    )

    return poly_modulus_degree


def find_max_precision(
    cfg: KeyParamsConfig, model: ONNXModel, poly_modulus_degree: int
) -> Tuple[int, int]:
    """Finds the maximum possible int and fractional precision for a given key config
    and fixed poly modulus degree. First increases the fractional precision as far as
    possible and then further increases the int precision.

    Args:
        cfg (KeyGenConfig): The key config.
        poly_modulus_degree (int): The selected poly modulus degree.

    Returns:
        Tuple[int, int]: _description_
    """
    max_bit_size_sum = _MAX_BIT_SIZE_SUM_BY_POLY_MODULUS_DEGREE[poly_modulus_degree]

    # now that we have an upper limit for the maximum bit size sum, the
    # precision for the fractional part is increased as far as possible
    n_bits_fractional_precision = (
        max_bit_size_sum - 2 * model.n_bits_integer_precision
    ) // (model.multiplication_depth + 2)
    logger.trace(
        f"increased fractional precision to {n_bits_fractional_precision} bits"
    )

    # if there are still some bits left, the precision for the integer
    # part is further increased
    n_bits_int_precision = (
        max_bit_size_sum
        - (model.multiplication_depth + 2) * n_bits_fractional_precision
    ) // 2
    logger.trace(f"increased int precision to {n_bits_int_precision} bits")

    # no single bit size can exceed 60 (the largest one is int + fractional precision)
    if n_bits_fractional_precision + n_bits_int_precision > 60:
        n_bits_decrease = n_bits_fractional_precision + n_bits_int_precision - 60
        n_bits_fractional_precision -= n_bits_decrease
        logger.trace(
            f"decreased fractional precision to {n_bits_fractional_precision} bits"
        )

    return n_bits_int_precision, n_bits_fractional_precision


def find_optimal_parameters(cfg: KeyParamsConfig, model: ONNXModel) -> KeyParams:
    """Find the optimal set of key parameters for a given config.

    Args:
        cfg (KeyParamsConfig): The key parameter config.

    Returns:
        KeyParams: The resulting key parameters.
    """
    if cfg.n_bits_fractional_precision + model.n_bits_integer_precision > 60:
        raise ValueError(
            "sum of integer and fractional precision must not exceed 60 bits! "
            f"integer precision: {model.n_bits_integer_precision} bits, "
            f"fractional precision: {cfg.n_bits_fractional_precision} bits."
        )

    poly_modulus_degree = find_min_poly_modulus_degree(cfg, model)

    n_bits_int_precision, n_bits_fractional_precision = find_max_precision(
        cfg, model, poly_modulus_degree
    )

    coeff_mod_bit_sizes = (
        [n_bits_fractional_precision + n_bits_int_precision]
        + [n_bits_fractional_precision] * model.multiplication_depth
        + [n_bits_fractional_precision + n_bits_int_precision]
    )

    return KeyParams(poly_modulus_degree, coeff_mod_bit_sizes)


def save_context(context: ts.Context, path: Path) -> None:
    """Saves a TenSEAL context including the secret key into the specified file and
    into another file (with an additional suffix ".pub") without the secret key.

    Args:
        context (ts.Context): The TenSEAL context to save.
        path (str): Path for storing the context including the secret key.
    """
    with open(path, "wb") as output_file:
        output_file.write(context.serialize(save_secret_key=True))

    with open(f"{path}.pub", "wb") as output_file:
        output_file.write(context.serialize(save_secret_key=False))


def load_context(path: Path) -> ts.Context:
    """Loads a TenSEAL context from specified file.

    Args:
        path (Path): Path of the context file to load.

    Returns:
        ts.Context: The loaded TenSEAL context.
    """
    with open(path, "rb") as input_file:
        return ts.Context.load(input_file.read())


def save_vector(vector: ts.CKKSVector, path: Path) -> None:
    """Saves a CKKS vector into the specified file.

    Args:
        vector (ts.CKKSVector): The vector to be saved.
        path (Path): Path for storing the vector.
    """
    with open(path, "wb") as output_file:
        output_file.write(vector.serialize())


def load_vector(context: ts.Context, path: Path) -> ts.CKKSVector:
    """Loads a CKKS vector from the specified file.

    Args:
        context (ts.Context): Context to be used for the loaded vector.
        path (Path): Path of the file containing the vector to load.

    Returns:
        ts.CKKSVector: The loaded vector.
    """
    with open(path, "rb") as input_file:
        return ts.ckks_vector_from(context, input_file.read())
