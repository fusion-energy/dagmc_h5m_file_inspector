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


def test_load_dagmc_data_decodes_sets_once(separated_boxes):
    """Opening the file once is not enough on its own, the readers also have to
    share the decoded sets rather than each rebuilding them."""
    filename = separated_boxes["filename"]

    with patch.object(core, "_read_sets_h5py", wraps=core._read_sets_h5py) as sets_mock:
        core._load_dagmc_data(filename, backend="h5py")

    assert sets_mock.call_count == 1


@pytest.mark.parametrize("tag_name", ["CATEGORY", "GEOM_DIMENSION", "NAME"])
def test_load_dagmc_data_decodes_each_tag_once(separated_boxes, tag_name):
    filename = separated_boxes["filename"]

    with patch.object(core, "_read_tag_h5py", wraps=core._read_tag_h5py) as tag_mock:
        core._load_dagmc_data(filename, backend="h5py")

    reads = [call.args[1] for call in tag_mock.call_args_list]
    assert reads.count(tag_name) == 1


def test_load_dagmc_data_reports_materials_without_volumes(
    empty_material_group_boxes,
):
    """The snapshot carries every mat: group, not just the ones in use."""
    filename = empty_material_group_boxes["filename"]

    data = core._load_dagmc_data(filename, backend="h5py")

    assert data.materials == empty_material_group_boxes["materials"]
    assert "mat_unused" not in data.volume_materials.values()
