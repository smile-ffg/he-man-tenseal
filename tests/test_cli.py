import pytest
from click.testing import CliRunner

import he_man_tenseal.main


@pytest.mark.parametrize("parameter, value", [("--something", 1), ("--path", "b")])
def test_invalid_parameters(parameter, value):
    """tests that the cli fails with wrong parameters"""
    runner = CliRunner()
    result = runner.invoke(he_man_tenseal.main.command_line, [parameter, value])
    assert result.exit_code != 0
