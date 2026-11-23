"""The stored notebooks: each can be opened in the browser, and each is stored executed and free of errors."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


def test_all_four_notebooks_are_there():
    assert [path.name[:2] for path in NOTEBOOKS] == [f"{i:02d}" for i in range(1, 5)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_opens_in_colab_and_is_stored_executed(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]

    # the badge under the title opens this very notebook, not a neighbour's
    title = "".join(cells[0]["source"])
    assert cells[0]["cell_type"] == "markdown" and title.startswith("# ")
    assert "colab.research.google.com/github/" in title and f"/notebooks/{path.name})" in title

    # the first code cell fetches the repository when the notebook runs on Colab, and only there
    setup = "".join(cells[1]["source"])
    assert cells[1]["cell_type"] == "code"
    assert 'if "google.colab" in sys.modules:' in setup and '"clone"' in setup

    # every other code cell was run, and none of them failed
    for cell in cells[2:]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None, "a code cell is stored without having been run"
            assert all(output["output_type"] != "error" for output in cell["outputs"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_links_between_notebooks_point_at_files_that_exist(path):
    text = " ".join("".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "markdown")
    for other in NOTEBOOKS:
        text = text.replace(f"]({other.name})", "")
    assert ".ipynb)" not in text.replace(f"/notebooks/{path.name})", ""), "a link to a notebook that is not there"
