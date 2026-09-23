from unittest.mock import patch

import pytest

from dagmc_h5m_file_inspector import core


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_load_dagmc_data_matches_public_readers(separated_boxes, backend):
    filename = separated_boxes["filename"]

    data = core._load_dagmc_data(filename, backend=backend)

    assert sorted(data.volume_data) == core.get_volumes(filename, backend=backend)
    assert data.volume_materials == core.get_volumes_and_materials(
        filename, backend=backend
    )


def test_load_dagmc_data_opens_h5py_file_once(separated_boxes):
    filename = separated_boxes["filename"]

    with patch.object(core.h5py, "File", wraps=core.h5py.File) as file_mock:
        core._load_dagmc_data(filename, backend="h5py")

    assert file_mock.call_count == 1


def test_load_dagmc_data_loads_pymoab_file_once(separated_boxes):
    filename = separated_boxes["filename"]

    with patch.object(core, "_load_moab_file", wraps=core._load_moab_file) as load_mock:
        core._load_dagmc_data(filename, backend="pymoab")

    assert load_mock.call_count == 1
