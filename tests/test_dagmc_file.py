from unittest.mock import patch

import numpy as np
import pytest

import dagmc_h5m_file_inspector as di
from dagmc_h5m_file_inspector import dagmc_file


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_queries_match_functional_api(touching_boxes, backend):
    filename = touching_boxes["filename"]
    model = di.DAGMCFile(filename, backend=backend)

    assert model.get_volumes() == di.get_volumes(filename, backend=backend)
    assert model.get_materials() == di.get_materials(filename, backend=backend)
    assert model.get_materials(remove_prefix=False) == di.get_materials(
        filename, remove_prefix=False, backend=backend
    )
    assert model.get_volumes_and_materials() == di.get_volumes_and_materials(
        filename, backend=backend
    )
    assert model.get_volumes_and_materials(
        remove_prefix=False
    ) == di.get_volumes_and_materials(filename, remove_prefix=False, backend=backend)

    expected_triangles = di.get_triangle_conn_and_coords_by_volume(
        filename, backend=backend
    )
    actual_triangles = model.get_triangle_conn_and_coords_by_volume()
    assert actual_triangles.keys() == expected_triangles.keys()
    for volume_id in actual_triangles:
        np.testing.assert_array_equal(
            actual_triangles[volume_id][0], expected_triangles[volume_id][0]
        )
        np.testing.assert_array_equal(
            actual_triangles[volume_id][1], expected_triangles[volume_id][1]
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_bounding_box_matches_functional_api(touching_boxes, backend):
    filename = touching_boxes["filename"]
    model = di.DAGMCFile(filename, backend=backend)

    actual = model.get_bounding_box()
    expected = di.get_bounding_box(filename, backend=backend)
    np.testing.assert_allclose(actual.lower_left, expected.lower_left)
    np.testing.assert_allclose(actual.upper_right, expected.upper_right)

    actual_material = model.get_bounding_box(materials="small_box")
    expected_material = di.get_bounding_box(
        filename, materials="small_box", backend=backend
    )
    np.testing.assert_allclose(actual_material.lower_left, expected_material.lower_left)
    np.testing.assert_allclose(
        actual_material.upper_right, expected_material.upper_right
    )


def test_dagmc_file_loads_once_for_multiple_queries(separated_boxes):
    filename = separated_boxes["filename"]

    with patch.object(
        dagmc_file, "_load_dagmc_data", wraps=dagmc_file._load_dagmc_data
    ) as load_mock:
        model = di.DAGMCFile(filename)
        model.get_volumes()
        model.get_materials()
        model.get_volumes_and_materials()
        model.get_bounding_box()
        model.get_triangle_conn_and_coords_by_volume()

    assert load_mock.call_count == 1


def test_dagmc_file_returns_triangle_data_copies(cube_geometry):
    model = di.DAGMCFile(cube_geometry["filename"])

    first = model.get_triangle_conn_and_coords_by_volume()
    first[1][0][0, 0] = -999
    first[1][1][0, 0] = -999.0
    second = model.get_triangle_conn_and_coords_by_volume()

    assert second[1][0][0, 0] != -999
    assert second[1][1][0, 0] != -999.0


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_mutates_in_memory_and_writes_once(
    touching_boxes, cube_geometry, backend, tmp_path
):
    combined = str(tmp_path / "combined.h5m")
    output = str(tmp_path / f"modified_{backend}.h5m")
    di.combine_h5m_files(
        [touching_boxes["filename"], cube_geometry["filename"]],
        combined,
        backend=backend,
    )

    with patch.object(
        dagmc_file, "_load_dagmc_data", wraps=dagmc_file._load_dagmc_data
    ) as load_mock:
        model = di.DAGMCFile(combined, backend=backend)
        assert model.remove_volumes(1) == [1]
        assert model.remove_materials("big_box") == ["big_box"]
        model.move(x=10.0)
        model.rotate_around_axis(axis="z", degrees=90)

        assert model.get_volumes_and_materials() == {3: "cube"}
        bounding_box = model.get_bounding_box()
        assert bounding_box.center == pytest.approx((0.0, 10.0, 0.0))
        assert not (tmp_path / f"modified_{backend}.h5m").exists()
        assert model.write(output) == output

    assert load_mock.call_count == 1
    assert di.get_volumes_and_materials(output, backend="h5py") == {3: "cube"}
    written_box = di.get_bounding_box(output)
    assert written_box.center == pytest.approx((0.0, 10.0, 0.0))


def test_dagmc_file_mutation_errors_leave_data_unchanged(separated_boxes):
    model = di.DAGMCFile(separated_boxes["filename"])
    expected = model.get_volumes_and_materials()

    with pytest.raises(ValueError, match="None of the specified volume IDs"):
        model.remove_volumes(999)
    with pytest.raises(ValueError, match="None of the specified materials"):
        model.remove_materials("missing")
    with pytest.raises(ValueError, match="Invalid axis"):
        model.rotate_around_axis(axis="invalid")

    assert model.get_volumes_and_materials() == expected


def test_dagmc_file_remove_multiple_values(separated_boxes):
    model = di.DAGMCFile(separated_boxes["filename"])

    assert model.remove_volumes([2, 1]) == [1, 2]
    assert model.get_volumes() == []
    assert model.get_materials() == []


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_missing_input_raises(backend):
    with pytest.raises(FileNotFoundError):
        di.DAGMCFile("does_not_exist.h5m", backend=backend)


def test_dagmc_file_invalid_backend_raises(cube_geometry):
    with pytest.raises(ValueError, match="Invalid backend"):
        di.DAGMCFile(cube_geometry["filename"], backend="invalid")
