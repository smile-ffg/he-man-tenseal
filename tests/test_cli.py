import pytest
from click.testing import CliRunner

import tenseal_inference.main


@pytest.mark.parametrize("parameter, value", [("--something", 1), ("--path", "b")])
def test_invalid_parameters(parameter, value):
    """tests that the cli fails with wrong parameters"""
    runner = CliRunner()
    result = runner.invoke(tenseal_inference.main.command_line, [parameter, value])
    assert result.exit_code != 0
