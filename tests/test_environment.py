"""environment.yml (conda) and pyproject.toml (pip) are two lists of the same packages: keep them in step."""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # in the standard library from Python 3.11 on

ROOT = Path(__file__).parent.parent
CONDA_NAMES = {"torch": "pytorch", "matplotlib": "matplotlib-base"}  # what conda-forge calls them


def conda_environment():
    """Channels, conda packages and pip lines of environment.yml, which uses only this much of YAML."""
    found = {"channels": [], "dependencies": [], "pip": []}
    section = None
    for line in (ROOT / "environment.yml").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            section = line.split(":")[0]
        elif line.strip() == "- pip:":
            section = "pip"
        elif line.strip().startswith("- "):
            found[section].append(line.strip()[2:])
    return found["channels"], found["dependencies"], found["pip"]


def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_the_conda_environment_holds_what_pip_would_install():
    requirements = project()["dependencies"] + sum(project()["optional-dependencies"].values(), [])
    wanted = set()
    for requirement in requirements:
        name = re.match(r"[A-Za-z0-9_.-]+", requirement).group()
        wanted.add(CONDA_NAMES.get(name, name) + requirement[len(name) :])
    _, packages, pip = conda_environment()
    assert {p for p in packages if p != "pip" and not p.startswith("python=")} == wanted
    assert "pip" in packages and pip == ["-e ."], "the library itself is installed from this folder, editable"


def test_the_conda_environment_takes_one_channel_and_a_python_the_library_supports():
    channels, packages, _ = conda_environment()
    assert channels == ["conda-forge", "nodefaults"]
    (python,) = [p for p in packages if p.startswith("python=")]
    oldest = re.fullmatch(r">=(\d+)\.(\d+)", project()["requires-python"])
    assert tuple(map(int, python.split("=")[1].split("."))) >= tuple(map(int, oldest.groups()))
