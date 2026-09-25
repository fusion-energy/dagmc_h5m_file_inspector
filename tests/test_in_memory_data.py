from unittest.mock import patch

import pytest

import dagmc_h5m_file_inspector as di
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
    """Reading the metadata takes a single pass over the file."""
    filename = separated_boxes["filename"]

    with patch.object(core.h5py, "File", wraps=core.h5py.File) as file_mock:
        data = core._load_dagmc_data(filename, backend="h5py")
        data.volume_materials  # noqa: B018 - reading it is what loads it

    assert file_mock.call_count == 1


def test_load_dagmc_data_loads_pymoab_file_once(separated_boxes):
    """Both readers share one Core, so needing all of the data still only
    loads the file once."""
    filename = separated_boxes["filename"]

    with patch.object(core, "_load_moab_file", wraps=core._load_moab_file) as load_mock:
        data = core._load_dagmc_data(filename, backend="pymoab")
        data.volume_materials  # noqa: B018 - reading it is what loads it
        data.volume_data  # noqa: B018 - and this brings in the triangles

    assert load_mock.call_count == 1


def test_load_dagmc_data_decodes_sets_once(separated_boxes):
    """Opening the file once is not enough on its own, the readers also have to
    share the decoded sets rather than each rebuilding them."""
    filename = separated_boxes["filename"]

    with patch.object(core, "_read_sets_h5py", wraps=core._read_sets_h5py) as sets_mock:
        data = core._load_dagmc_data(filename, backend="h5py")
        data.volume_materials  # noqa: B018 - reading it is what loads it

    assert sets_mock.call_count == 1


@pytest.mark.parametrize("tag_name", ["CATEGORY", "GEOM_DIMENSION", "NAME"])
def test_load_dagmc_data_decodes_each_tag_once(separated_boxes, tag_name):
    filename = separated_boxes["filename"]

    with patch.object(core, "_read_tag_h5py", wraps=core._read_tag_h5py) as tag_mock:
        data = core._load_dagmc_data(filename, backend="h5py")
        data.volume_materials  # noqa: B018 - reading it is what loads it

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


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_load_dagmc_data_defers_reading_triangles(separated_boxes, backend):
    data = core._load_dagmc_data(separated_boxes["filename"], backend=backend)

    assert data.metadata_loaded is False
    assert data.volume_data_loaded is False
    assert data.volume_ids == separated_boxes["volumes"]
    assert data.metadata_loaded is True

    data.volume_data  # noqa: B018 - reading the property is what loads it
    assert data.volume_data_loaded is True


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_ids_match_the_triangle_data_keys(separated_boxes, backend):
    """The ids are read from the sets and the triangle data is read later, so
    the two have to agree about which volumes exist."""
    data = core._load_dagmc_data(separated_boxes["filename"], backend=backend)

    assert sorted(data.volume_ids) == sorted(data.volume_data)


@pytest.mark.parametrize(
    "query",
    ["get_volumes", "get_materials", "get_volumes_and_materials"],
)
@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_metadata_queries_do_not_read_triangles(separated_boxes, backend, query):
    """These answers come from the sets and tags, so asking for them should not
    pay for the triangle data."""
    model = di.DAGMCFile(separated_boxes["filename"], backend=backend)
    assert model._data.volume_data_loaded is False

    getattr(model, query)()

    assert model._data.volume_data_loaded is False


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_geometry_queries_do_read_triangles(separated_boxes, backend):
    model = di.DAGMCFile(separated_boxes["filename"], backend=backend)

    model.get_triangle_conn_and_coords_by_volume()

    assert model._data.volume_data_loaded is True


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_unfiltered_bounding_box_does_not_read_triangles(separated_boxes, backend):
    """The box over everything comes from the coordinate block, so it does not
    need the triangles split up per volume."""
    model = di.DAGMCFile(separated_boxes["filename"], backend=backend)

    model.get_bounding_box()

    assert model._data.volume_data_loaded is False


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_filtered_bounding_box_reads_triangles(separated_boxes, backend):
    model = di.DAGMCFile(separated_boxes["filename"], backend=backend)

    model.get_bounding_box(materials=separated_boxes["materials"][0])

    assert model._data.volume_data_loaded is True


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_uses_memory_once_mutated(separated_boxes, backend):
    """After a move the file on disk no longer describes the geometry, so the
    box has to come from the in-memory coordinates."""
    model = di.DAGMCFile(separated_boxes["filename"], backend=backend)
    before = model.get_bounding_box()

    model.move(x=100.0)
    after = model.get_bounding_box()

    assert after.lower_left[0] == pytest.approx(before.lower_left[0] + 100.0)
    assert after.upper_right[0] == pytest.approx(before.upper_right[0] + 100.0)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_get_volumes_tracks_removals_without_the_triangle_data(
    separated_boxes, backend
):
    model = di.DAGMCFile(separated_boxes["filename"], backend=backend)

    assert model.remove_volumes(1) == [1]
    assert model.get_volumes() == [2]
    assert model.get_volumes() == sorted(model._data.volume_data)
