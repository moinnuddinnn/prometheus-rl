import tomllib
from pathlib import Path

import pytest

from prometheus_rl import __version__
from prometheus_rl.__main__ import main


def test_version_matches_metadata():
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert tomllib.loads(project.read_text())["project"]["version"] == __version__


def test_cli_status(capsys):
    assert main([]) == 0
    assert "No agents implemented yet" in capsys.readouterr().out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_unknown_argument(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--unknown"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
