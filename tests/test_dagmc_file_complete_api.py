from types import SimpleNamespace

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


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_sets_openmc_material_volumes(touching_boxes, backend):
    model = di.DAGMCFile(touching_boxes["filename"], backend=backend)
    materials = [
        SimpleNamespace(name="small_box", volume=None),
        SimpleNamespace(name="big_box", volume=None),
    ]

    model.set_openmc_material_volumes(materials)

    expected = model.get_volumes_by_material_name()
    assert materials[0].volume == pytest.approx(expected["small_box"])
    assert materials[1].volume == pytest.approx(expected["big_box"])


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_converts_to_vtkhdf(cube_geometry, backend, tmp_path):
    model = di.DAGMCFile(cube_geometry["filename"], backend=backend)
    output = str(tmp_path / f"cube_{backend}.vtkhdf")

    assert model.convert_to_vtkhdf(output) == output
    assert (tmp_path / f"cube_{backend}.vtkhdf").is_file()


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_sets_boundary_condition(cube_geometry, backend, tmp_path):
    model = di.DAGMCFile(cube_geometry["filename"], backend=backend)
    output = str(tmp_path / f"cube_bc_{backend}.h5m")

    assert model.set_boundary_condition(1, "vacuum", output) == output
    assert di.get_volumes(output, backend=backend) == [1]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_dagmc_file_combines_files(separated_boxes, cube_geometry, backend, tmp_path):
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


@pytest.mark.parametrize(
    "source_operation",
    [
        lambda model: model.get_surface_ids(),
        lambda model: model.get_volumes_by_cell_id(),
        lambda model: model.get_surface_area_by_surface_id(),
        lambda model: model.convert_to_vtkhdf(),
        lambda model: model.set_boundary_condition(1, "vacuum"),
    ],
    ids=["surface-ids", "volume-sizes", "surface-areas", "convert", "boundary"],
)
def test_source_file_operations_reject_modified_data(cube_geometry, source_operation):
    model = di.DAGMCFile(cube_geometry["filename"])
    model.move(x=10.0)

    with pytest.raises(RuntimeError, match="Write the file and load the output"):
        source_operation(model)
