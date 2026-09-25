"""Behaviour across geometries with varied volume-to-material combinations.

The fixtures used here cover several volumes sharing one material, one material
per volume, an uneven spread, a volume in no material group at all, and a
material group holding no volumes.
"""

import numpy as np
import pytest

import dagmc_h5m_file_inspector as di

BACKENDS = ["h5py", "pymoab"]

# fixtures whose volumes all sit in a mat: group, so they round trip cleanly
WRITABLE_FIXTURES = [
    "one_material_boxes",
    "two_material_boxes",
    "three_material_boxes",
    "uneven_material_boxes",
    "empty_material_group_boxes",
]

ALL_FIXTURES = WRITABLE_FIXTURES + ["orphan_volume_boxes"]


@pytest.fixture
def geometry(request):
    """Resolve a fixture name from parametrisation to its fixture value."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("geometry", ALL_FIXTURES, indirect=True)
@pytest.mark.parametrize("backend", BACKENDS)
def test_volumes_and_materials_match_the_geometry(geometry, backend):
    model = di.DAGMCFile(geometry["filename"], backend=backend)

    assert model.get_volumes() == geometry["volumes"]
    assert model.get_materials() == geometry["materials"]
    assert model.get_materials(remove_prefix=False) == geometry["materials_with_prefix"]
    assert model.get_volumes_and_materials() == geometry["volumes_and_materials"]
    assert (
        model.get_volumes_and_materials(remove_prefix=False)
        == geometry["volumes_and_materials_with_prefix"]
    )


@pytest.mark.parametrize("geometry", ALL_FIXTURES, indirect=True)
@pytest.mark.parametrize("backend", BACKENDS)
def test_class_api_agrees_with_functional_api(geometry, backend):
    """The README steers users from the functions to the class, so the two
    must not answer differently on any of these geometries."""
    filename = geometry["filename"]
    model = di.DAGMCFile(filename, backend=backend)

    assert model.get_volumes() == di.get_volumes(filename, backend=backend)
    assert model.get_materials() == di.get_materials(filename, backend=backend)
    assert model.get_materials(remove_prefix=False) == di.get_materials(
        filename, remove_prefix=False, backend=backend
    )
    assert model.get_volumes_and_materials() == di.get_volumes_and_materials(
        filename, backend=backend
    )


@pytest.mark.parametrize("geometry", ALL_FIXTURES, indirect=True)
def test_both_backends_agree(geometry):
    """A file should read the same whichever backend opens it."""
    filename = geometry["filename"]
    h5py_model = di.DAGMCFile(filename, backend="h5py")
    pymoab_model = di.DAGMCFile(filename, backend="pymoab")

    assert h5py_model.get_volumes() == pymoab_model.get_volumes()
    assert h5py_model.get_materials() == pymoab_model.get_materials()
    assert (
        h5py_model.get_volumes_and_materials()
        == pymoab_model.get_volumes_and_materials()
    )


@pytest.mark.parametrize("geometry", ALL_FIXTURES, indirect=True)
@pytest.mark.parametrize("backend", BACKENDS)
def test_materials_with_no_volumes_are_still_reported(geometry, backend):
    """An empty mat: group is in the file, so it belongs in get_materials even
    though no volume maps to it."""
    model = di.DAGMCFile(geometry["filename"], backend=backend)
    unassigned = geometry.get("materials_without_volumes", [])

    for material in unassigned:
        assert material in model.get_materials()
        assert material not in model.get_volumes_and_materials().values()


@pytest.mark.parametrize(
    "geometry",
    ["one_material_boxes", "two_material_boxes", "uneven_material_boxes"],
    indirect=True,
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_volume_sizes_sum_per_shared_material(geometry, backend):
    """Several volumes in one group should sum, not overwrite each other."""
    model = di.DAGMCFile(geometry["filename"], backend=backend)
    sizes = model.get_volumes_by_material_name()

    for material, volume_ids in geometry["volumes_by_material"].items():
        expected = sum(geometry["expected_volume_sizes"][v] for v in volume_ids)
        assert sizes[material] == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("geometry", WRITABLE_FIXTURES, indirect=True)
@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_covers_every_volume(geometry, backend):
    model = di.DAGMCFile(geometry["filename"], backend=backend)
    bbox = model.get_bounding_box()

    assert np.allclose(bbox.lower_left, geometry["lower_left"], atol=0.01)
    assert np.allclose(bbox.upper_right, geometry["upper_right"], atol=0.01)


@pytest.mark.parametrize("backend", BACKENDS)
def test_removing_a_shared_material_removes_all_its_volumes(
    uneven_material_boxes, backend
):
    """mat_a covers volumes 1, 2 and 4 so all three should go at once."""
    model = di.DAGMCFile(uneven_material_boxes["filename"], backend=backend)

    assert model.remove_materials("mat_a") == ["mat_a"]
    assert model.get_volumes() == [3, 5]
    assert model.get_materials() == ["mat_b", "mat_c"]
    assert model.get_volumes_and_materials() == {3: "mat_b", 5: "mat_c"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_removing_some_volumes_of_a_material_keeps_the_material(
    uneven_material_boxes, backend
):
    model = di.DAGMCFile(uneven_material_boxes["filename"], backend=backend)

    assert model.remove_volumes([1, 2]) == [1, 2]
    # volume 4 is still mat_a, so the material stays
    assert model.get_materials() == ["mat_a", "mat_b", "mat_c"]

    assert model.remove_volumes(4) == [4]
    # its last volume has gone now, so mat_a goes with it
    assert model.get_materials() == ["mat_b", "mat_c"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_material_with_no_volumes_survives_unrelated_removals(
    empty_material_group_boxes, backend
):
    model = di.DAGMCFile(empty_material_group_boxes["filename"], backend=backend)

    model.remove_volumes(1)
    assert "mat_unused" in model.get_materials()

    assert model.remove_materials("mat_unused") == ["mat_unused"]
    assert "mat_unused" not in model.get_materials()


@pytest.mark.parametrize("geometry", WRITABLE_FIXTURES, indirect=True)
@pytest.mark.parametrize("backend", BACKENDS)
def test_write_round_trips_volumes_and_materials(geometry, backend, tmp_path):
    model = di.DAGMCFile(geometry["filename"], backend=backend)
    output = str(tmp_path / "round_trip.h5m")
    model.write(output)

    reloaded = di.DAGMCFile(output, backend=backend)
    assert reloaded.get_volumes() == geometry["volumes"]
    assert reloaded.get_volumes_and_materials() == geometry["volumes_and_materials"]
    # an empty mat: group has no volumes to write, so it does not survive
    assert reloaded.get_materials() == sorted(
        set(geometry["volumes_and_materials"].values())
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_volume_outside_any_material_group_is_read_but_not_written(
    orphan_volume_boxes, backend, tmp_path
):
    """A volume in no mat: group cannot be written, so it should fail with the
    offending ids rather than a KeyError from inside the writer."""
    model = di.DAGMCFile(orphan_volume_boxes["filename"], backend=backend)

    assert model.get_volumes() == orphan_volume_boxes["volumes"]
    assert (
        model.get_volumes_and_materials()
        == orphan_volume_boxes["volumes_and_materials"]
    )

    with pytest.raises(ValueError, match="have no material group"):
        model.write(str(tmp_path / "orphan.h5m"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_removing_the_orphan_volume_makes_the_file_writable(
    orphan_volume_boxes, backend, tmp_path
):
    model = di.DAGMCFile(orphan_volume_boxes["filename"], backend=backend)
    orphan = orphan_volume_boxes["orphan_volumes"][0]

    assert model.remove_volumes(orphan) == [orphan]

    output = str(tmp_path / "cleaned.h5m")
    model.write(output)
    reloaded = di.DAGMCFile(output, backend=backend)
    assert reloaded.get_volumes() == orphan_volume_boxes["assigned_volumes"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_ignores_a_material_with_no_volumes(
    empty_material_group_boxes, backend
):
    """Asking for a material that has no volumes is an error, not a KeyError."""
    model = di.DAGMCFile(empty_material_group_boxes["filename"], backend=backend)

    with pytest.raises(ValueError, match="No volumes found for materials"):
        model.get_bounding_box(materials="mat_unused")
