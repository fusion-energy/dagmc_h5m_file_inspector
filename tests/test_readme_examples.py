import re
import shutil
from pathlib import Path

import pytest

from dagmc_h5m_file_inspector import DAGMCFile
from dagmc_h5m_file_inspector.core import _load_dagmc_data, _write_h5m

README = Path(__file__).parents[1] / "README.md"
PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _readme_python_blocks():
    """Return each README Python fence with a descriptive source name."""
    text = README.read_text(encoding="utf-8")
    matches = list(PYTHON_BLOCK.finditer(text))
    if len(matches) != text.count("```python"):
        raise RuntimeError("Every README Python fence must contain executable code")

    blocks = []
    for number, match in enumerate(matches, start=1):
        headings = list(HEADING.finditer(text, 0, match.start()))
        section = headings[-1].group(1) if headings else "README"
        blocks.append(
            pytest.param(
                match.group(1),
                id=f"{number:02d}-{section.lower().replace(' ', '-')}",
            )
        )
    return blocks


@pytest.fixture(scope="session")
def readme_example_files(
    tmp_path_factory,
    grouped_boxes,
    cube_geometry,
    sphere_geometry,
    rectangle_geometry,
):
    """Create the named input files referenced by README examples."""
    directory = tmp_path_factory.mktemp("readme_examples")
    shutil.copy2(grouped_boxes["filename"], directory / "dagmc.h5m")
    shutil.copy2(cube_geometry["filename"], directory / "file_a.h5m")
    shutil.copy2(sphere_geometry["filename"], directory / "file_b.h5m")

    reactor_source = directory / "reactor_source.h5m"
    DAGMCFile.combine_h5m_files(
        [
            cube_geometry["filename"],
            sphere_geometry["filename"],
            rectangle_geometry["filename"],
        ],
        str(reactor_source),
    )
    reactor_data = _load_dagmc_data(str(reactor_source))
    _write_h5m(
        str(directory / "reactor.h5m"),
        reactor_data.volume_data,
        {1: "blanket", 2: "first_wall", 3: "shield"},
    )
    _write_h5m(
        str(directory / "volume_removal.h5m"),
        reactor_data.volume_data,
        {1: "steel", 2: "steel", 3: "water"},
    )
    reactor_source.unlink()
    return directory


@pytest.mark.parametrize("code", _readme_python_blocks())
def test_readme_python_examples_execute(
    code,
    readme_example_files,
    tmp_path,
    monkeypatch,
):
    """Execute each README Python fence directly in an isolated directory."""
    for source in readme_example_files.iterdir():
        shutil.copy2(source, tmp_path / source.name)

    if "import openmc" in code:
        pytest.importorskip("openmc")

    monkeypatch.chdir(tmp_path)
    compiled = compile(code, str(README), "exec")
    exec(compiled, {"__name__": "__main__"})
