from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import dagmc_h5m_file_inspector as di


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_group_queries_match_functional_api(grouped_boxes, backend):
    filename = grouped_boxes["filename"]
    model = di.DAGMCFile(filename, backend=backend)

    assert model.get_cell_ids_by_group_name() == di.get_cell_ids_by_group_name(
        filename, backend=backend
    )
    assert model.get_groups_by_cell_id() == di.get_groups_by_cell_id(
        filename, backend=backend
    )
    assert model.get_cell_ids_by_group_name() == grouped_boxes["cell_ids_by_group_name"]
    assert model.get_groups_by_cell_id() == grouped_boxes["groups_by_cell_id"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_geometry_queries_match_functional_api(touching_boxes, backend):
    filename = touching_boxes["filename"]
    model = di.DAGMCFile(filename, backend=backend)

    assert model.get_surface_ids() == di.get_surface_ids(filename, backend=backend)
    assert model.get_surface_ids_by_cell_id(1) == di.get_surface_ids_by_cell_id(
        filename, cell_id=1, backend=backend
    )
    assert model.get_surface_ids_by_material_name(
        "small_box"
    ) == di.get_surface_ids_by_material_name(
        filename, material="small_box", backend=backend
    )
    assert model.get_volumes_by_cell_id() == pytest.approx(
        di.get_volumes_by_cell_id(filename, backend=backend)
    )
    assert model.get_volumes_by_material_name() == pytest.approx(
        di.get_volumes_by_material_name(filename, backend=backend)
    )
    assert model.get_volumes_by_cell_id_and_material_name() == pytest.approx(
        di.get_volumes_by_cell_id_and_material_name(filename, backend=backend)
    )
    assert model.get_surface_area_by_cell_id(1) == pytest.approx(
        di.get_surface_area_by_cell_id(filename, cell_id=1, backend=backend)
    )
    assert model.get_surface_area_by_material_name("small_box") == pytest.approx(
        di.get_surface_area_by_material_name(
            filename, material="small_box", backend=backend
        )
    )
    assert model.get_surface_area_by_surface_id() == pytest.approx(
        di.get_surface_area_by_surface_id(filename, backend=backend)
    )
    assert model.get_surface_shared_status() == di.get_surface_shared_status(
        filename, backend=backend
    )
    assert model.get_volumes_by_cell_id() == pytest.approx(
        {1: 1000.0, 2: 8000.0}, rel=1e-10, abs=1e-9
    )
    assert model.get_volumes_by_material_name() == pytest.approx(
        {"small_box": 1000.0, "big_box": 8000.0}, rel=1e-10, abs=1e-9
    )
    assert sorted(model.get_surface_area_by_cell_id(1)) == pytest.approx([100.0] * 6)
    assert sum(model.get_surface_area_by_cell_id(2)) == pytest.approx(2400.0)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_sets_openmc_material_volumes(touching_boxes, backend):
    model = di.DAGMCFile(touching_boxes["filename"], backend=backend)
    materials = [
        SimpleNamespace(name="small_box", volume=None),
        SimpleNamespace(name="big_box", volume=None),
    ]

    model.set_openmc_material_volumes(materials)

    assert materials[0].volume == pytest.approx(1000.0, rel=1e-10, abs=1e-9)
    assert materials[1].volume == pytest.approx(8000.0, rel=1e-10, abs=1e-9)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_converts_to_vtkhdf(
    cube_geometry, backend, tmp_path, assert_closed_mesh
):
    model = di.DAGMCFile(cube_geometry["filename"], backend=backend)
    output = str(tmp_path / f"cube_{backend}.vtkhdf")

    assert model.convert_to_vtkhdf(output) == output
    assert (tmp_path / f"cube_{backend}.vtkhdf").is_file()
    with h5py.File(output, "r") as converted:
        mesh = converted["VTKHDF"]
        assert_closed_mesh(
            mesh["Connectivity"][()].reshape(-1, 3), mesh["Points"][()], 1000.0, 600.0
        )
        np.testing.assert_array_equal(mesh["CellData/cell_id"][()], 1)
        np.testing.assert_array_equal(mesh["CellData/material_id"][()], 0)
        assert mesh["FieldData/material_names"].asstr()[()].tolist() == ["cube"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_sets_boundary_condition(
    cube_geometry, backend, tmp_path, boundary_group_members
):
    model = di.DAGMCFile(cube_geometry["filename"], backend=backend)
    output = str(tmp_path / f"cube_bc_{backend}.h5m")

    assert model.set_boundary_condition(1, "vacuum", output) == output
    assert di.get_volumes(output, backend=backend) == [1]
    assert boundary_group_members(output) == {"boundary:vacuum": [("Surface", 1)]}


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_combines_files(
    separated_boxes, cube_geometry, backend, tmp_path, assert_closed_mesh
):
    output = str(tmp_path / f"combined_{backend}.h5m")

    combined = di.DAGMCFile.combine_h5m_files(
        [separated_boxes["filename"], cube_geometry["filename"]],
        output_file=output,
        backend=backend,
    )

    assert isinstance(combined, di.DAGMCFile)
    assert combined.filename == output
    assert combined.backend == backend
    assert combined.get_volumes() == [1, 2, 3]
    assert combined.get_materials() == ["box_a", "box_b", "cube"]
    assert combined.get_volumes_by_cell_id() == pytest.approx(
        {1: 1000.0, 2: 1000.0, 3: 1000.0}, rel=1e-10, abs=1e-9
    )
    for volume_id, (
        connectivity,
        coordinates,
    ) in combined.get_triangle_conn_and_coords_by_volume().items():
        assert_closed_mesh(connectivity, coordinates, 1000.0, 600.0)
        expected_center = (15.0, 0.0, 0.0) if volume_id == 2 else (0.0, 0.0, 0.0)
        np.testing.assert_allclose(
            (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2.0,
            expected_center,
            atol=1e-9,
        )


@pytest.mark.parametrize(
    "source_operation",
    [
        lambda model: model.get_surface_ids(),
        lambda model: model.get_volumes_by_cell_id(),
        lambda model: model.get_surface_area_by_surface_id(),
        lambda model: model.convert_to_vtkhdf(),
        lambda model: model.set_boundary_condition(1, "vacuum"),
        lambda model: model.get_surface_ids_by_cell_id(1),
        lambda model: model.get_surface_ids_by_material_name("cube"),
        lambda model: model.get_surface_area_by_cell_id(1),
        lambda model: model.get_surface_area_by_material_name("cube"),
        lambda model: model.get_surface_shared_status(),
        lambda model: model.get_volumes_by_material_name(),
        lambda model: model.get_volumes_by_cell_id_and_material_name(),
        lambda model: model.get_cell_ids_by_group_name(),
        lambda model: model.get_groups_by_cell_id(),
        lambda model: model.set_openmc_material_volumes([]),
    ],
    ids=[
        "surface-ids",
        "volume-sizes",
        "surface-areas",
        "convert",
        "boundary",
        "cell-surfaces",
        "material-surfaces",
        "cell-areas",
        "material-areas",
        "shared-surfaces",
        "material-volumes",
        "cell-material-volumes",
        "group-cells",
        "cell-groups",
        "openmc-volumes",
    ],
)
@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_source_file_operations_reject_modified_data(
    cube_geometry, source_operation, backend
):
    model = di.DAGMCFile(cube_geometry["filename"], backend=backend)
    model.move(x=10.0)

    with pytest.raises(RuntimeError, match="Write the file and load the output"):
        source_operation(model)
