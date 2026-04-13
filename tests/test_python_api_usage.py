import os

import h5py
import numpy as np
import pytest

import dagmc_h5m_file_inspector as di


def test_version():
    assert isinstance(di.__version__, str)
    assert len(di.__version__) > 0


# ============================================================================
# Tests for touching boxes geometry
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_and_material_extraction_without_stripped_prefix(
    touching_boxes, backend
):
    """Extracts the volume numbers and material ids from a dagmc file and
    checks the contents match the expected contents"""

    dict_of_vol_and_mats = di.get_volumes_and_materials(
        filename=touching_boxes["filename"],
        remove_prefix=False,
        backend=backend,
    )

    assert dict_of_vol_and_mats == touching_boxes["volumes_and_materials_with_prefix"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_and_material_extraction_remove_prefix(touching_boxes, backend):
    """Extracts the volume numbers and material ids from a dagmc file and
    checks the contents match the expected contents"""

    dict_of_vol_and_mats = di.get_volumes_and_materials(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    assert dict_of_vol_and_mats == touching_boxes["volumes_and_materials"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_extraction(touching_boxes, backend):
    """Extracts the volume ids from a dagmc file and checks the contents
    match the expected contents"""

    volumes = di.get_volumes(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    assert volumes == touching_boxes["volumes"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_extraction(touching_boxes, backend):
    """Extracts the surface ids from a dagmc file and checks that
    surface ids are returned as a sorted list of integers"""

    surfaces = di.get_surface_ids(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    assert isinstance(surfaces, list)
    assert len(surfaces) > 0
    assert all(isinstance(s, int) for s in surfaces)
    assert surfaces == sorted(surfaces)
    # two cuboids: 6 surfaces each = 12 surface meshsets
    assert len(surfaces) == 12


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_extraction_separated_boxes(separated_boxes, backend):
    """Extracts the surface ids from separated boxes and checks that
    surface ids are returned correctly"""

    surfaces = di.get_surface_ids(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    assert isinstance(surfaces, list)
    assert len(surfaces) > 0
    assert all(isinstance(s, int) for s in surfaces)
    assert surfaces == sorted(surfaces)
    # two cuboids: 6 surfaces each = 12 surface meshsets
    assert len(surfaces) == 12


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_extraction_cube(cube_geometry, backend):
    """A cube should have 6 surfaces"""

    surfaces = di.get_surface_ids(
        filename=cube_geometry["filename"],
        backend=backend,
    )

    assert len(surfaces) == cube_geometry["expected_num_surfaces"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_extraction_cylinder(cylinder_geometry, backend):
    """A cylinder should have 3 surfaces (top cap, bottom cap, lateral)"""

    surfaces = di.get_surface_ids(
        filename=cylinder_geometry["filename"],
        backend=backend,
    )

    assert len(surfaces) == cylinder_geometry["expected_num_surfaces"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_extraction_file_not_found(backend):
    """Checks that a FileNotFoundError is raised for missing files"""

    with pytest.raises(FileNotFoundError):
        di.get_surface_ids(
            filename="non_existant.h5m",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_filter_openmc_transport(touching_boxes, backend, tmp_path):
    """Verify that surface IDs from get_surface_ids can be used in an
    OpenMC SurfaceFilter tally with DAGMC geometry."""
    import openmc

    filename = touching_boxes["filename"]

    # Get surface IDs using the inspector
    surface_ids = di.get_surface_ids(filename=filename, backend=backend)
    assert len(surface_ids) > 0

    # Set up cross sections (H1 only)
    xs_path = os.path.join(os.path.dirname(__file__), "ENDFB-7.1-NNDC_H1.h5")
    xs_xml = str(tmp_path / "cross_sections.xml")
    with open(xs_xml, "w") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<cross_sections>\n"
            f'  <library materials="H1" path="{xs_path}" type="neutron"/>\n'
            "</cross_sections>\n"
        )
    openmc.config["cross_sections"] = xs_xml

    # Create materials matching the DAGMC file
    mat1 = openmc.Material(name="small_box")
    mat1.add_nuclide("H1", 1.0, "ao")
    mat1.set_density("g/cm3", 0.001)
    mat2 = openmc.Material(name="big_box")
    mat2.add_nuclide("H1", 1.0, "ao")
    mat2.set_density("g/cm3", 0.001)
    materials = openmc.Materials([mat1, mat2])

    # DAGMC geometry
    dag_univ = openmc.DAGMCUniverse(filename=filename)
    bound_dag_univ = dag_univ.bounded_universe()
    geometry = openmc.Geometry(root=bound_dag_univ)

    # Point source near center of small_box
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0.1, 0.1, 0.1))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14e6], [1])

    settings = openmc.Settings()
    settings.batches = 2
    settings.particles = 1000
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.source = source

    # Create a surface filter tally using the discovered surface IDs
    surface_filter = openmc.SurfaceFilter(surface_ids)
    tally = openmc.Tally(name="surface_current")
    tally.filters = [surface_filter]
    tally.scores = ["current"]

    model = openmc.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=openmc.Tallies([tally]),
    )

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        output_file = model.run(output=False)
        sp = openmc.StatePoint(output_file)
        tally_result = sp.get_tally(name="surface_current")
        current = tally_result.mean.flatten()
        # At least some surfaces should have non-zero current
        assert current.sum() > 0
        # The tally should have one bin per surface
        assert len(current) == len(surface_ids)
    finally:
        os.chdir(original_dir)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_material_extraction_no_remove_prefix(touching_boxes, backend):
    """Extracts the materials tags from a dagmc file and checks the
    contents match the expected contents"""

    materials = di.get_materials(
        filename=touching_boxes["filename"],
        remove_prefix=False,
        backend=backend,
    )

    assert materials == touching_boxes["materials_with_prefix"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_material_extraction_remove_prefix(touching_boxes, backend):
    """Extracts the materials tags from a dagmc file and checks the
    contents match the expected contents"""

    materials = di.get_materials(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    assert materials == touching_boxes["materials"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_fail_with_missing_input_files(backend):
    """Calls functions without necessary input files to check if error
    handling is working"""

    with pytest.raises(FileNotFoundError):
        di.get_volumes_and_materials(
            filename="non_existant.h5m",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box(touching_boxes, backend):
    """Extracts the bounding box from a dagmc file and checks it matches
    the expected geometry bounds"""

    bb = di.get_bounding_box(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    np.testing.assert_allclose(bb.lower_left, touching_boxes["lower_left"], rtol=1e-5)
    np.testing.assert_allclose(bb.upper_right, touching_boxes["upper_right"], rtol=1e-5)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_single_material(touching_boxes, backend):
    """Bounding box for a single material (small_box) should match that
    volume's geometry only"""

    bb = di.get_bounding_box(
        filename=touching_boxes["filename"],
        materials="small_box",
        backend=backend,
    )

    np.testing.assert_allclose(
        bb.lower_left, touching_boxes["small_box_lower_left"], rtol=1e-5
    )
    np.testing.assert_allclose(
        bb.upper_right, touching_boxes["small_box_upper_right"], rtol=1e-5
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_single_material_big(touching_boxes, backend):
    """Bounding box for a single material (big_box) should match that
    volume's geometry only"""

    bb = di.get_bounding_box(
        filename=touching_boxes["filename"],
        materials="big_box",
        backend=backend,
    )

    np.testing.assert_allclose(
        bb.lower_left, touching_boxes["big_box_lower_left"], rtol=1e-5
    )
    np.testing.assert_allclose(
        bb.upper_right, touching_boxes["big_box_upper_right"], rtol=1e-5
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_material_list(touching_boxes, backend):
    """Bounding box for all materials as a list should match the global
    bounding box"""

    bb = di.get_bounding_box(
        filename=touching_boxes["filename"],
        materials=["small_box", "big_box"],
        backend=backend,
    )

    np.testing.assert_allclose(bb.lower_left, touching_boxes["lower_left"], rtol=1e-5)
    np.testing.assert_allclose(bb.upper_right, touching_boxes["upper_right"], rtol=1e-5)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_single_material_string(touching_boxes, backend):
    """Passing a single string should give the same result as a single-element list"""

    bb_str = di.get_bounding_box(
        filename=touching_boxes["filename"],
        materials="small_box",
        backend=backend,
    )

    bb_list = di.get_bounding_box(
        filename=touching_boxes["filename"],
        materials=["small_box"],
        backend=backend,
    )

    np.testing.assert_allclose(bb_str.lower_left, bb_list.lower_left)
    np.testing.assert_allclose(bb_str.upper_right, bb_list.upper_right)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_invalid_material(touching_boxes, backend):
    """Passing a material name not in the file should raise ValueError"""

    with pytest.raises(ValueError):
        di.get_bounding_box(
            filename=touching_boxes["filename"],
            materials="nonexistent",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_returns_bounding_box_type(touching_boxes, backend):
    """The return type should be BoundingBox"""

    bb = di.get_bounding_box(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    assert isinstance(bb, di.BoundingBox)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box_properties(touching_boxes, backend):
    """BoundingBox properties should return correct values"""

    bb = di.get_bounding_box(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    np.testing.assert_allclose(bb.lower_left, touching_boxes["lower_left"], rtol=1e-5)
    np.testing.assert_allclose(bb.upper_right, touching_boxes["upper_right"], rtol=1e-5)

    expected_center = (
        touching_boxes["lower_left"] + touching_boxes["upper_right"]
    ) / 2.0
    np.testing.assert_allclose(bb.center, expected_center, rtol=1e-5)

    expected_width = touching_boxes["upper_right"] - touching_boxes["lower_left"]
    np.testing.assert_allclose(bb.width, expected_width, rtol=1e-5)

    expected_volume = float(np.prod(expected_width))
    assert abs(bb.volume - expected_volume) / expected_volume < 1e-5


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_sizes_by_cell_id(touching_boxes, backend):
    """Extracts the geometric volumes from a dagmc file and checks they
    match the expected cube volumes"""

    volume_sizes = di.get_volumes_by_cell_id(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    expected = touching_boxes["expected_volume_sizes"]

    for vol_id, expected_size in expected.items():
        assert vol_id in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[vol_id] - expected_size) / expected_size < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_sizes_by_material_name(touching_boxes, backend):
    """Extracts the geometric volumes by material name from a dagmc file"""

    volume_sizes = di.get_volumes_by_material_name(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # small_box is volume 1 (1000), big_box is volume 2 (8000)
    expected = {
        "small_box": touching_boxes["expected_volume_sizes"][1],
        "big_box": touching_boxes["expected_volume_sizes"][2],
    }

    for mat_name, expected_size in expected.items():
        assert mat_name in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[mat_name] - expected_size) / expected_size < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_sizes_by_cell_id_and_material_name(touching_boxes, backend):
    """Extracts the geometric volumes by cell ID and material name from a dagmc file"""

    volume_sizes = di.get_volumes_by_cell_id_and_material_name(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # small_box is volume 1 (1000), big_box is volume 2 (8000)
    expected = {
        (1, "small_box"): touching_boxes["expected_volume_sizes"][1],
        (2, "big_box"): touching_boxes["expected_volume_sizes"][2],
    }

    for key, expected_size in expected.items():
        assert key in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[key] - expected_size) / expected_size < 0.05


# ============================================================================
# Tests for separated boxes geometry
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_and_material_extraction(separated_boxes, backend):
    """Extracts the volume numbers and material ids from separated boxes"""

    dict_of_vol_and_mats = di.get_volumes_and_materials(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    assert dict_of_vol_and_mats == separated_boxes["volumes_and_materials"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_extraction(separated_boxes, backend):
    """Extracts the volume ids from separated boxes"""

    volumes = di.get_volumes(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    assert volumes == separated_boxes["volumes"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_material_extraction(separated_boxes, backend):
    """Extracts the materials tags from separated boxes"""

    materials = di.get_materials(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    assert materials == separated_boxes["materials"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_bounding_box(separated_boxes, backend):
    """Extracts the bounding box from separated boxes"""

    bb = di.get_bounding_box(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    np.testing.assert_allclose(bb.lower_left, separated_boxes["lower_left"], rtol=1e-5)
    np.testing.assert_allclose(
        bb.upper_right, separated_boxes["upper_right"], rtol=1e-5
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_bounding_box_single_material(separated_boxes, backend):
    """Bounding box for a single material (box_a) in separated geometry"""

    bb = di.get_bounding_box(
        filename=separated_boxes["filename"],
        materials="box_a",
        backend=backend,
    )

    np.testing.assert_allclose(
        bb.lower_left, separated_boxes["box_a_lower_left"], rtol=1e-5
    )
    np.testing.assert_allclose(
        bb.upper_right, separated_boxes["box_a_upper_right"], rtol=1e-5
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_bounding_box_material_list(separated_boxes, backend):
    """Bounding box for all materials as a list should match the global
    bounding box for separated geometry"""

    bb = di.get_bounding_box(
        filename=separated_boxes["filename"],
        materials=["box_a", "box_b"],
        backend=backend,
    )

    np.testing.assert_allclose(bb.lower_left, separated_boxes["lower_left"], rtol=1e-5)
    np.testing.assert_allclose(
        bb.upper_right, separated_boxes["upper_right"], rtol=1e-5
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_sizes_by_cell_id(separated_boxes, backend):
    """Extracts the geometric volumes from separated boxes and checks they
    match the expected cube volumes"""

    volume_sizes = di.get_volumes_by_cell_id(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    expected = separated_boxes["expected_volume_sizes"]

    for vol_id, expected_size in expected.items():
        assert vol_id in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[vol_id] - expected_size) / expected_size < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_sizes_by_material_name(separated_boxes, backend):
    """Extracts the geometric volumes by material name from separated boxes"""

    volume_sizes = di.get_volumes_by_material_name(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    # box_a is volume 1, box_b is volume 2
    expected = {
        "box_a": separated_boxes["expected_volume_sizes"][1],
        "box_b": separated_boxes["expected_volume_sizes"][2],
    }

    for mat_name, expected_size in expected.items():
        assert mat_name in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[mat_name] - expected_size) / expected_size < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_sizes_by_cell_id_and_material_name(separated_boxes, backend):
    """Extracts geometric volumes by cell ID and material name from separated boxes."""

    volume_sizes = di.get_volumes_by_cell_id_and_material_name(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    # box_a is volume 1, box_b is volume 2
    expected = {
        (1, "box_a"): separated_boxes["expected_volume_sizes"][1],
        (2, "box_b"): separated_boxes["expected_volume_sizes"][2],
    }

    for key, expected_size in expected.items():
        assert key in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[key] - expected_size) / expected_size < 0.05


# ============================================================================
# Tests for h5py and pymoab backend consistency
# ============================================================================

# All h5m test files
H5M_TEST_FILES = [
    "tests/circulartorus.h5m",
    "tests/cuboid.h5m",
    "tests/cylinder.h5m",
    "tests/ellipticaltorus.h5m",
    "tests/nestedcylinder.h5m",
    "tests/nestedsphere.h5m",
    "tests/oktavian.h5m",
    "tests/simpletokamak.h5m",
    "tests/sphere.h5m",
    "tests/tetrahedral.h5m",
    "tests/two_tetrahedrons.h5m",
    "tests/twotouchingcuboids.h5m",
]


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volume_ids_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return the same volume IDs"""

    h5py_volumes = di.get_volumes(filename, backend="h5py")
    pymoab_volumes = di.get_volumes(filename, backend="pymoab")

    assert h5py_volumes == pymoab_volumes, (
        f"Volume IDs differ: h5py={h5py_volumes}, pymoab={pymoab_volumes}"
    )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_surface_ids_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return the same surface IDs"""

    h5py_surfaces = di.get_surface_ids(filename, backend="h5py")
    pymoab_surfaces = di.get_surface_ids(filename, backend="pymoab")

    assert h5py_surfaces == pymoab_surfaces, (
        f"Surface IDs differ: h5py={h5py_surfaces}, pymoab={pymoab_surfaces}"
    )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_material_tags_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return the same material tags"""

    h5py_materials = di.get_materials(filename, backend="h5py")
    pymoab_materials = di.get_materials(filename, backend="pymoab")

    assert h5py_materials == pymoab_materials, (
        f"Material tags differ: h5py={h5py_materials}, pymoab={pymoab_materials}"
    )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volumes_and_materials_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return the same volume-to-material mapping"""

    h5py_mapping = di.get_volumes_and_materials(filename, backend="h5py")
    pymoab_mapping = di.get_volumes_and_materials(filename, backend="pymoab")

    assert h5py_mapping == pymoab_mapping, (
        f"Volume-material mapping differs: h5py={h5py_mapping}, pymoab={pymoab_mapping}"
    )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volume_sizes_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends produce the same volume calculations"""

    h5py_volumes = di.get_volumes_by_cell_id(filename, backend="h5py")
    pymoab_volumes = di.get_volumes_by_cell_id(filename, backend="pymoab")

    # Check same volume IDs are returned
    assert set(h5py_volumes.keys()) == set(pymoab_volumes.keys()), (
        f"Volume IDs differ: h5py={set(h5py_volumes.keys())}, "
        f"pymoab={set(pymoab_volumes.keys())}"
    )

    # Check volumes match within tolerance
    for vol_id in h5py_volumes:
        h5py_vol = h5py_volumes[vol_id]
        pymoab_vol = pymoab_volumes[vol_id]

        # Use relative tolerance for non-zero volumes
        if pymoab_vol > 1e-10:
            rel_diff = abs(h5py_vol - pymoab_vol) / pymoab_vol
            assert rel_diff < 0.01, (
                f"Volume {vol_id} differs: h5py={h5py_vol}, "
                f"pymoab={pymoab_vol}, rel_diff={rel_diff}"
            )
        else:
            # For near-zero volumes, use absolute tolerance
            assert abs(h5py_vol - pymoab_vol) < 1e-6, (
                f"Volume {vol_id} differs: h5py={h5py_vol}, pymoab={pymoab_vol}"
            )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volume_sizes_by_material_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab produce the same volume calculations by material."""

    h5py_volumes = di.get_volumes_by_material_name(filename, backend="h5py")
    pymoab_volumes = di.get_volumes_by_material_name(filename, backend="pymoab")

    # Check same material names are returned
    assert set(h5py_volumes.keys()) == set(pymoab_volumes.keys()), (
        f"Material names differ: h5py={set(h5py_volumes.keys())}, "
        f"pymoab={set(pymoab_volumes.keys())}"
    )

    # Check volumes match within tolerance
    for mat_name in h5py_volumes:
        h5py_vol = h5py_volumes[mat_name]
        pymoab_vol = pymoab_volumes[mat_name]

        # Use relative tolerance for non-zero volumes
        if pymoab_vol > 1e-10:
            rel_diff = abs(h5py_vol - pymoab_vol) / pymoab_vol
            assert rel_diff < 0.01, (
                f"Material '{mat_name}' differs: h5py={h5py_vol}, "
                f"pymoab={pymoab_vol}, rel_diff={rel_diff}"
            )
        else:
            # For near-zero volumes, use absolute tolerance
            assert abs(h5py_vol - pymoab_vol) < 1e-6, (
                f"Material '{mat_name}' differs: h5py={h5py_vol}, pymoab={pymoab_vol}"
            )


# ============================================================================
# Tests for OpenMC material volume assignment
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_with_list(touching_boxes, backend):
    """Tests setting volumes on a list of OpenMC materials"""
    import openmc

    # Create OpenMC materials matching the DAGMC material names
    small_box_mat = openmc.Material(name="small_box")
    big_box_mat = openmc.Material(name="big_box")
    materials = [small_box_mat, big_box_mat]

    # Initially volumes should be None
    assert small_box_mat.volume is None
    assert big_box_mat.volume is None

    # Set volumes from DAGMC file
    di.set_openmc_material_volumes(
        materials=materials,
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # Check volumes are set correctly (with 5% tolerance for mesh discretization)
    expected = touching_boxes["expected_volume_sizes"]
    # small_box is volume 1, big_box is volume 2
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05
    assert abs(big_box_mat.volume - expected[2]) / expected[2] < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_with_materials_object(touching_boxes, backend):
    """Tests setting volumes on an OpenMC Materials collection"""
    import openmc

    # Create OpenMC materials and add to Materials collection
    small_box_mat = openmc.Material(name="small_box")
    big_box_mat = openmc.Material(name="big_box")
    materials = openmc.Materials([small_box_mat, big_box_mat])

    # Set volumes from DAGMC file
    di.set_openmc_material_volumes(
        materials=materials,
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # Check volumes are set correctly
    expected = touching_boxes["expected_volume_sizes"]
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05
    assert abs(big_box_mat.volume - expected[2]) / expected[2] < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_non_matching_materials(touching_boxes, backend):
    """Tests that materials without matching names are not affected"""
    import openmc

    # Create materials - one matching, one not
    small_box_mat = openmc.Material(name="small_box")
    unmatched_mat = openmc.Material(name="nonexistent_material")
    materials = [small_box_mat, unmatched_mat]

    # Set volumes from DAGMC file
    di.set_openmc_material_volumes(
        materials=materials,
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # Matching material should have volume set
    expected = touching_boxes["expected_volume_sizes"]
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05

    # Non-matching material should remain None
    assert unmatched_mat.volume is None


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_duplicate_names_error(touching_boxes, backend):
    """Tests that duplicate material names raise an error"""
    import openmc

    # Create materials with duplicate names
    mat1 = openmc.Material(name="small_box")
    mat2 = openmc.Material(name="small_box")  # Duplicate!
    materials = [mat1, mat2]

    # Should raise ValueError for duplicate names
    with pytest.raises(
        ValueError, match="Multiple OpenMC materials have the same name"
    ):
        di.set_openmc_material_volumes(
            materials=materials,
            filename=touching_boxes["filename"],
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_file_not_found(backend):
    """Tests that missing file raises FileNotFoundError"""
    import openmc

    mat = openmc.Material(name="test")
    materials = [mat]

    with pytest.raises(FileNotFoundError):
        di.set_openmc_material_volumes(
            materials=materials,
            filename="nonexistent_file.h5m",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_with_none_names(touching_boxes, backend):
    """Tests that materials with None names are ignored without error"""
    import openmc

    # Create materials - one with name, one without
    small_box_mat = openmc.Material(name="small_box")
    unnamed_mat = openmc.Material()  # No name (defaults to None)
    materials = [small_box_mat, unnamed_mat]

    # Should not raise error, unnamed materials are skipped
    di.set_openmc_material_volumes(
        materials=materials,
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # Named material should have volume set
    expected = touching_boxes["expected_volume_sizes"]
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05

    # Unnamed material should remain unchanged
    assert unnamed_mat.volume is None


# ============================================================================
# Tests comparing volume calculations with OpenMC stochastic
# ============================================================================


# Subset of files for OpenMC stochastic tests (faster execution)
H5M_TEST_FILES_OPENMC_STOCHASTIC = [
    "tests/cuboid.h5m",
    "tests/sphere.h5m",
    "tests/nestedsphere.h5m",
    "tests/cylinder.h5m",
]


@pytest.mark.parametrize("filename", H5M_TEST_FILES_OPENMC_STOCHASTIC)
def test_volume_sizes_openmc_stochastic_consistency(filename, tmp_path):
    """Verify our volume calculations match OpenMC stochastic results.

    OpenMC uses Monte Carlo sampling to estimate volumes, so we allow
    a larger tolerance (5%) to account for statistical noise.
    """
    from pathlib import Path

    import openmc

    # Convert to absolute path before changing directories
    abs_filename = str(Path(filename).resolve())

    # Set up cross sections (H1 only)
    xs_path = os.path.join(os.path.dirname(__file__), "ENDFB-7.1-NNDC_H1.h5")
    xs_xml = str(tmp_path / "cross_sections.xml")
    with open(xs_xml, "w") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<cross_sections>\n"
            f'  <library materials="H1" path="{xs_path}" type="neutron"/>\n'
            "</cross_sections>\n"
        )
    openmc.config["cross_sections"] = xs_xml

    # Get volumes and materials from our implementation
    bb = di.get_bounding_box(abs_filename)
    materials_list = di.get_materials(abs_filename, remove_prefix=True)

    # Create OpenMC materials matching the DAGMC file
    openmc_mats = []
    for mat_name in materials_list:
        mat = openmc.Material(name=mat_name)
        mat.add_nuclide("H1", 1.0)
        mat.set_density("g/cm3", 1.0)
        openmc_mats.append(mat)
    materials = openmc.Materials(openmc_mats)

    # Create DAGMC universe and geometry
    dagmc_univ = openmc.DAGMCUniverse(abs_filename, auto_geom_ids=True)
    bounded_univ = dagmc_univ.bounded_universe()
    geometry = openmc.Geometry(bounded_univ)

    # Settings for volume calculation
    settings = openmc.Settings()
    settings.run_mode = "volume"

    # Create volume calculation for all materials
    vol_calc = openmc.VolumeCalculation(
        domains=openmc_mats,
        samples=50000,
        lower_left=list(bb.lower_left),
        upper_right=list(bb.upper_right),
    )
    settings.volume_calculations = [vol_calc]

    # Build and run model
    model = openmc.Model(geometry=geometry, materials=materials, settings=settings)

    # Change to temp directory to avoid polluting the test directory
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        model.run(output=False)

        # Read results
        results = openmc.VolumeCalculation.from_hdf5("volume_1.h5")

        # Get OpenMC volumes by material name
        openmc_volumes_by_mat = {}
        for domain, vol in results.volumes.items():
            # domain is the material ID
            for mat in openmc_mats:
                if mat.id == domain:
                    openmc_volumes_by_mat[mat.name] = vol.nominal_value
                    break

        # Get our volumes by material name for comparison
        h5py_volumes_by_mat = di.get_volumes_by_material_name(
            abs_filename, backend="h5py"
        )

        # Compare volumes (allow 5% tolerance for stochastic noise)
        for mat_name in h5py_volumes_by_mat:
            if mat_name in openmc_volumes_by_mat:
                h5py_vol = h5py_volumes_by_mat[mat_name]
                openmc_vol = openmc_volumes_by_mat[mat_name]

                if openmc_vol > 1e-10:
                    rel_diff = abs(h5py_vol - openmc_vol) / openmc_vol
                    assert rel_diff < 0.05, (
                        f"Material '{mat_name}' volume differs: "
                        f"h5py={h5py_vol}, openmc={openmc_vol}, "
                        f"rel_diff={rel_diff}"
                    )
    finally:
        os.chdir(original_dir)


# ============================================================================
# Tests for get_triangle_conn_and_coords_by_volume
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_triangle_conn_and_coords_basic(touching_boxes, backend):
    """Test that triangle connectivity and coordinates are returned for each volume"""

    data = di.get_triangle_conn_and_coords_by_volume(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # Should have data for all expected volumes
    assert set(data.keys()) == set(touching_boxes["volumes"])

    for vol_id in touching_boxes["volumes"]:
        connectivity, coordinates = data[vol_id]

        # Connectivity should be Mx3 array of integers
        assert connectivity.ndim == 2
        assert connectivity.shape[1] == 3
        assert connectivity.dtype in [np.int64, np.int32, np.uint64]

        # Coordinates should be Nx3 array of floats
        assert coordinates.ndim == 2
        assert coordinates.shape[1] == 3
        assert coordinates.dtype == np.float64

        # All connectivity indices should be valid
        assert connectivity.min() >= 0
        assert connectivity.max() < len(coordinates)

        # Should have at least some triangles (boxes have 12 triangles minimum)
        assert len(connectivity) >= 12


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_triangle_conn_and_coords_separated_boxes(separated_boxes, backend):
    """Test triangle connectivity and coordinates for separated boxes geometry"""

    data = di.get_triangle_conn_and_coords_by_volume(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    # Should have data for all expected volumes
    assert set(data.keys()) == set(separated_boxes["volumes"])

    for vol_id in separated_boxes["volumes"]:
        connectivity, coordinates = data[vol_id]

        # Basic shape checks
        assert connectivity.ndim == 2
        assert connectivity.shape[1] == 3
        assert coordinates.ndim == 2
        assert coordinates.shape[1] == 3

        # Connectivity indices should be valid
        assert connectivity.min() >= 0
        assert connectivity.max() < len(coordinates)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_triangle_conn_and_coords_file_not_found(backend):
    """Test that missing file raises FileNotFoundError"""

    with pytest.raises(FileNotFoundError):
        di.get_triangle_conn_and_coords_by_volume(
            filename="nonexistent_file.h5m",
            backend=backend,
        )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_triangle_conn_and_coords_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return equivalent triangle data"""

    h5py_data = di.get_triangle_conn_and_coords_by_volume(filename, backend="h5py")
    pymoab_data = di.get_triangle_conn_and_coords_by_volume(filename, backend="pymoab")

    # Same volume IDs should be returned
    assert set(h5py_data.keys()) == set(pymoab_data.keys()), (
        f"Volume IDs differ: h5py={set(h5py_data.keys())}, "
        f"pymoab={set(pymoab_data.keys())}"
    )

    for vol_id in h5py_data:
        h5py_conn, h5py_coords = h5py_data[vol_id]
        pymoab_conn, pymoab_coords = pymoab_data[vol_id]

        # Same number of triangles
        assert len(h5py_conn) == len(pymoab_conn), (
            f"Volume {vol_id}: triangle count differs - "
            f"h5py={len(h5py_conn)}, pymoab={len(pymoab_conn)}"
        )

        # Same number of unique vertices
        assert len(h5py_coords) == len(pymoab_coords), (
            f"Volume {vol_id}: vertex count differs - "
            f"h5py={len(h5py_coords)}, pymoab={len(pymoab_coords)}"
        )

        # Coordinates should be the same (may be in different order)
        # Sort coordinates for comparison
        h5py_sorted = np.sort(h5py_coords, axis=0)
        pymoab_sorted = np.sort(pymoab_coords, axis=0)
        np.testing.assert_allclose(
            h5py_sorted,
            pymoab_sorted,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Volume {vol_id}: coordinates differ",
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_triangle_conn_and_coords_mesh_validity(touching_boxes, backend):
    """Test that the returned mesh data can be used to create valid PyVista meshes"""

    data = di.get_triangle_conn_and_coords_by_volume(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    try:
        import pyvista as pv

        HAS_PYVISTA = True
    except ImportError:
        HAS_PYVISTA = False

    if not HAS_PYVISTA:
        pytest.skip("pyvista not installed")

    for vol_id in touching_boxes["volumes"]:
        connectivity, coordinates = data[vol_id]

        # Convert to PyVista format (prepend 3 to each row)
        n_triangles = connectivity.shape[0]
        faces = np.hstack(
            [np.full((n_triangles, 1), 3, dtype=np.int64), connectivity]
        ).flatten()

        # Create PyVista mesh - this should not raise an error
        mesh = pv.PolyData(coordinates, faces)

        # Mesh should have the correct number of cells (triangles)
        assert mesh.n_cells == n_triangles

        # Mesh should have the correct number of points
        assert mesh.n_points == len(coordinates)


# ============================================================================
# Tests for convert_h5m_to_vtkhdf
# ============================================================================


def test_convert_h5m_to_vtkhdf_structure(touching_boxes, tmp_path):
    """Test that the output VTKHDF file has the correct HDF5 structure"""

    output_file = str(tmp_path / "output.vtkhdf")
    result = di.convert_h5m_to_vtkhdf(
        h5m_filename=touching_boxes["filename"],
        vtkhdf_filename=output_file,
    )

    assert result == output_file
    assert os.path.isfile(output_file)

    with h5py.File(output_file, "r") as f:
        # Root group and attributes
        assert "VTKHDF" in f
        root = f["VTKHDF"]
        assert list(root.attrs["Version"]) == [2, 1]
        type_val = root.attrs["Type"]
        if isinstance(type_val, bytes):
            type_val = type_val.decode("ascii")
        assert type_val == "UnstructuredGrid"

        # Required datasets
        for ds_name in [
            "NumberOfPoints",
            "NumberOfCells",
            "NumberOfConnectivityIds",
            "Points",
            "Connectivity",
            "Offsets",
            "Types",
        ]:
            assert ds_name in root, f"Missing dataset: {ds_name}"

        # CellData group
        assert "CellData" in root
        assert "cell_id" in root["CellData"]
        assert "material_id" in root["CellData"]

        # FieldData group
        assert "FieldData" in root
        assert "material_names" in root["FieldData"]


def test_convert_h5m_to_vtkhdf_data_integrity(touching_boxes, tmp_path):
    """Test that the VTKHDF file contains correct mesh data"""

    output_file = str(tmp_path / "output.vtkhdf")
    di.convert_h5m_to_vtkhdf(
        h5m_filename=touching_boxes["filename"],
        vtkhdf_filename=output_file,
    )

    # Get expected data from the source
    per_vol = di.get_triangle_conn_and_coords_by_volume(
        filename=touching_boxes["filename"],
    )
    expected_n_tris = sum(len(c) for c, _ in per_vol.values())
    expected_n_verts = sum(len(v) for _, v in per_vol.values())

    with h5py.File(output_file, "r") as f:
        root = f["VTKHDF"]

        n_points = root["NumberOfPoints"][0]
        n_cells = root["NumberOfCells"][0]
        n_conn_ids = root["NumberOfConnectivityIds"][0]

        assert n_points == expected_n_verts
        assert n_cells == expected_n_tris
        assert n_conn_ids == expected_n_tris * 3

        assert root["Points"].shape == (expected_n_verts, 3)
        assert root["Connectivity"].shape == (expected_n_tris * 3,)
        assert root["Offsets"].shape == (expected_n_tris + 1,)
        assert root["Types"].shape == (expected_n_tris,)

        # All types should be VTK_TRIANGLE = 5
        assert np.all(root["Types"][()] == 5)

        # Offsets should be [0, 3, 6, 9, ...]
        expected_offsets = np.arange(0, expected_n_tris * 3 + 1, 3, dtype=np.int64)
        np.testing.assert_array_equal(root["Offsets"][()], expected_offsets)

        # Connectivity indices should be valid
        conn = root["Connectivity"][()]
        assert np.all(conn >= 0)
        assert np.all(conn < n_points)


def test_convert_h5m_to_vtkhdf_cell_data(touching_boxes, tmp_path):
    """Test that cell_id and material_id arrays are correct"""

    output_file = str(tmp_path / "output.vtkhdf")
    di.convert_h5m_to_vtkhdf(
        h5m_filename=touching_boxes["filename"],
        vtkhdf_filename=output_file,
    )

    vol_mat = di.get_volumes_and_materials(
        filename=touching_boxes["filename"],
    )
    per_vol = di.get_triangle_conn_and_coords_by_volume(
        filename=touching_boxes["filename"],
    )

    with h5py.File(output_file, "r") as f:
        root = f["VTKHDF"]
        cell_ids = root["CellData/cell_id"][()]
        material_ids = root["CellData/material_id"][()]
        n_cells = root["NumberOfCells"][0]

        assert len(cell_ids) == n_cells
        assert len(material_ids) == n_cells

        # Check that cell_ids contain the expected volume IDs
        unique_cell_ids = sorted(set(cell_ids.tolist()))
        assert unique_cell_ids == sorted(per_vol.keys())

        # Check that the number of triangles per volume matches
        for vol_id in per_vol:
            expected_count = len(per_vol[vol_id][0])
            actual_count = np.sum(cell_ids == vol_id)
            assert actual_count == expected_count

        # Check material_names in FieldData
        mat_names = [
            v.decode("utf-8") if isinstance(v, bytes) else v
            for v in root["FieldData/material_names"][()]
        ]
        unique_materials = sorted(set(vol_mat.values()))
        assert mat_names == unique_materials

        # Check material_ids are consistent with cell_ids and the mapping
        mat_to_int = {name: idx for idx, name in enumerate(unique_materials)}
        for vol_id, mat_name in vol_mat.items():
            mask = cell_ids == vol_id
            if np.any(mask):
                expected_mat_id = mat_to_int[mat_name]
                assert np.all(material_ids[mask] == expected_mat_id)


def test_convert_h5m_to_vtkhdf_default_filename(touching_boxes, tmp_path):
    """Test that omitting vtkhdf_filename produces correct default"""

    import shutil

    # Copy h5m into tmp_path so the output goes there
    src = touching_boxes["filename"]
    dst = str(tmp_path / "dagmc.h5m")
    shutil.copy2(src, dst)

    result = di.convert_h5m_to_vtkhdf(h5m_filename=dst)

    expected = str(tmp_path / "dagmc.vtkhdf")
    assert result == expected
    assert os.path.isfile(expected)


def test_convert_h5m_to_vtkhdf_file_not_found():
    """Test that missing input file raises FileNotFoundError"""

    with pytest.raises(FileNotFoundError):
        di.convert_h5m_to_vtkhdf(h5m_filename="nonexistent.h5m")


# ============================================================================
# Tests for remove_materials
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_remove_single_material(touching_boxes, backend, tmp_path):
    """Remove one material from touching geometry, verify the other remains."""
    input_file = touching_boxes["filename"]

    # Check materials and volumes before removal
    mats_before = di.get_materials(input_file, backend="pymoab")
    vols_before = di.get_volumes(input_file, backend="pymoab")
    assert "small_box" in mats_before
    assert "big_box" in mats_before

    output = str(tmp_path / f"removed_{backend}.h5m")
    removed = di.remove_materials(
        input_filename=input_file,
        output_filename=output,
        materials_to_remove="small_box",
        backend=backend,
    )
    assert removed == ["small_box"]

    mats_after = di.get_materials(output, backend="pymoab")
    assert mats_after == ["big_box"]

    vols_after = di.get_volumes(output, backend="pymoab")
    assert len(vols_after) == 1

    assert len(mats_after) < len(mats_before)
    assert len(vols_after) < len(vols_before)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_remove_multiple_materials(separated_boxes, backend, tmp_path):
    """Remove all materials, verify empty result."""
    output = str(tmp_path / f"removed_all_{backend}.h5m")
    removed = di.remove_materials(
        input_filename=separated_boxes["filename"],
        output_filename=output,
        materials_to_remove=["box_a", "box_b"],
        backend=backend,
    )
    assert removed == ["box_a", "box_b"]

    # Use h5py to read (empty files may not be loadable by pymoab)
    mats = di.get_materials(output, backend="h5py")
    assert mats == []


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_remove_material_string_input(separated_boxes, backend, tmp_path):
    """Single string accepted for materials_to_remove."""
    output = str(tmp_path / f"string_{backend}.h5m")
    removed = di.remove_materials(
        input_filename=separated_boxes["filename"],
        output_filename=output,
        materials_to_remove="box_b",
        backend=backend,
    )
    assert removed == ["box_b"]
    mats = di.get_materials(output, backend="pymoab")
    assert mats == ["box_a"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_remove_nonexistent_material_raises(separated_boxes, backend, tmp_path):
    """ValueError when material not found."""
    output = str(tmp_path / f"nope_{backend}.h5m")
    with pytest.raises(ValueError, match="None of the specified materials"):
        di.remove_materials(
            input_filename=separated_boxes["filename"],
            output_filename=output,
            materials_to_remove="nonexistent",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_remove_material_file_not_found(backend, tmp_path):
    """FileNotFoundError for missing input."""
    output = str(tmp_path / "out.h5m")
    with pytest.raises(FileNotFoundError):
        di.remove_materials(
            input_filename="does_not_exist.h5m",
            output_filename=output,
            materials_to_remove="mat",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_input_file_not_modified(separated_boxes, backend, tmp_path):
    """Original file unchanged after removal."""
    import hashlib

    input_file = separated_boxes["filename"]
    with open(input_file, "rb") as fh:
        hash_before = hashlib.md5(fh.read()).hexdigest()

    output = str(tmp_path / f"modified_check_{backend}.h5m")
    di.remove_materials(
        input_filename=input_file,
        output_filename=output,
        materials_to_remove="box_a",
        backend=backend,
    )

    with open(input_file, "rb") as fh:
        hash_after = hashlib.md5(fh.read()).hexdigest()

    assert hash_before == hash_after


def test_remove_materials_h5py_pymoab_consistency(touching_boxes, tmp_path):
    """Both backends produce files with same materials and volumes."""
    output_h5py = str(tmp_path / "h5py_out.h5m")
    output_pymoab = str(tmp_path / "pymoab_out.h5m")

    di.remove_materials(
        input_filename=touching_boxes["filename"],
        output_filename=output_h5py,
        materials_to_remove="small_box",
        backend="h5py",
    )
    di.remove_materials(
        input_filename=touching_boxes["filename"],
        output_filename=output_pymoab,
        materials_to_remove="small_box",
        backend="pymoab",
    )

    # Read each output with pymoab (which can read both formats)
    mats_h5py = di.get_materials(output_h5py, backend="pymoab")
    mats_pymoab = di.get_materials(output_pymoab, backend="pymoab")
    assert mats_h5py == mats_pymoab

    vols_h5py = di.get_volumes(output_h5py, backend="pymoab")
    vols_pymoab = di.get_volumes(output_pymoab, backend="pymoab")
    assert len(vols_h5py) == len(vols_pymoab)


def test_output_readable_by_both_backends(separated_boxes, tmp_path):
    """Output from h5py backend readable by pymoab and vice versa."""
    output_h5py = str(tmp_path / "from_h5py.h5m")
    di.remove_materials(
        input_filename=separated_boxes["filename"],
        output_filename=output_h5py,
        materials_to_remove="box_a",
        backend="h5py",
    )

    # h5py-written output should be readable by both backends
    mats_via_h5py = di.get_materials(output_h5py, backend="h5py")
    mats_via_pymoab = di.get_materials(output_h5py, backend="pymoab")
    assert mats_via_h5py == mats_via_pymoab

    output_pymoab = str(tmp_path / "from_pymoab.h5m")
    di.remove_materials(
        input_filename=separated_boxes["filename"],
        output_filename=output_pymoab,
        materials_to_remove="box_a",
        backend="pymoab",
    )

    # pymoab-written output should be readable by pymoab
    mats_via_pymoab2 = di.get_materials(output_pymoab, backend="pymoab")
    assert mats_via_pymoab2 == ["box_b"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_remove_material_openmc_transport(touching_boxes, backend, tmp_path):
    """Verify that the h5m file produced by remove_materials is a
    valid DAGMC geometry by running OpenMC fixed-source particle transport
    through it.
    """
    import openmc

    output = str(tmp_path / f"transport_{backend}.h5m")
    di.remove_materials(
        input_filename=touching_boxes["filename"],
        output_filename=output,
        materials_to_remove="small_box",
        backend=backend,
    )

    # Set up cross sections (H1 only)
    xs_path = os.path.join(os.path.dirname(__file__), "ENDFB-7.1-NNDC_H1.h5")
    xs_xml = str(tmp_path / "cross_sections.xml")
    with open(xs_xml, "w") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<cross_sections>\n"
            f'  <library materials="H1" path="{xs_path}" type="neutron"/>\n'
            "</cross_sections>\n"
        )
    openmc.config["cross_sections"] = xs_xml

    # Create material matching remaining "big_box"
    mat = openmc.Material(name="big_box")
    mat.add_nuclide("H1", 1.0, "ao")
    mat.set_density("g/cm3", 0.001)
    materials = openmc.Materials([mat])

    # DAGMC geometry
    dag_univ = openmc.DAGMCUniverse(filename=output)
    bound_dag_univ = dag_univ.bounded_universe()
    geometry = openmc.Geometry(root=bound_dag_univ)

    # Point source near center of big_box
    bb = di.get_bounding_box(output, materials="big_box")
    center = bb.center
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(
        (center[0] + 0.1, center[1] + 0.1, center[2] + 0.1)
    )
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14e6], [1])

    settings = openmc.Settings()
    settings.batches = 2
    settings.particles = 1000
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.source = source

    tally = openmc.Tally(name="flux")
    tally.scores = ["flux"]

    model = openmc.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=openmc.Tallies([tally]),
    )

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        output_file = model.run(output=False)
        sp = openmc.StatePoint(output_file)
        flux = sp.get_tally(name="flux").mean.flatten()[0]
        # Flux should be positive (particles traversed the geometry)
        assert flux > 0
    finally:
        os.chdir(original_dir)


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_convert_h5m_to_vtkhdf_all_geometries(filename, tmp_path):
    """Test conversion works for all test geometries"""

    output_file = str(tmp_path / "output.vtkhdf")
    result = di.convert_h5m_to_vtkhdf(
        h5m_filename=filename,
        vtkhdf_filename=output_file,
    )

    assert os.path.isfile(result)

    per_vol = di.get_triangle_conn_and_coords_by_volume(filename=filename)
    expected_n_tris = sum(len(c) for c, _ in per_vol.values() if len(c) > 0)

    with h5py.File(output_file, "r") as f:
        root = f["VTKHDF"]
        assert root["NumberOfCells"][0] == expected_n_tris
        assert root["CellData/cell_id"].shape == (expected_n_tris,)
        assert root["CellData/material_id"].shape == (expected_n_tris,)


# ============================================================================
# Tests for surface area functions
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_cell_id_cube(cube_geometry, backend):
    """Test surface areas of a 10x10x10 cube by cell ID.

    Planar faces are exact after triangulation, so we use a tight tolerance.
    """
    areas = di.get_surface_area_by_cell_id(
        filename=cube_geometry["filename"],
        cell_id=cube_geometry["cell_id"],
        backend=backend,
    )
    assert len(areas) == cube_geometry["expected_num_surfaces"]
    for area in sorted(areas):
        assert area == pytest.approx(
            cube_geometry["expected_surface_area_each"], rel=1e-10
        )
    assert sum(areas) == pytest.approx(
        cube_geometry["expected_total_surface_area"], rel=1e-10
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_cell_id_sphere(sphere_geometry, backend):
    """Test surface area of a sphere with radius 5 by cell ID.

    Curved surfaces lose area to mesh discretization (~2%), so 5% tolerance.
    """
    areas = di.get_surface_area_by_cell_id(
        filename=sphere_geometry["filename"],
        cell_id=sphere_geometry["cell_id"],
        backend=backend,
    )
    assert len(areas) >= 1
    assert sum(areas) == pytest.approx(
        sphere_geometry["expected_total_surface_area"], rel=0.05
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_cell_id_cylinder(cylinder_geometry, backend):
    """Test surface areas of a cylinder (h=20, r=5) by cell ID.

    3 surfaces: two caps and one lateral. The lateral surface is curved, and
    the caps are planar but have circular boundaries that lose area to chord
    discretization (~4%), so all surfaces use 5% tolerance.
    """
    areas = di.get_surface_area_by_cell_id(
        filename=cylinder_geometry["filename"],
        cell_id=cylinder_geometry["cell_id"],
        backend=backend,
    )
    assert len(areas) == cylinder_geometry["expected_num_surfaces"]
    sorted_areas = sorted(areas)
    # Two smallest are the caps (pi*r^2 each)
    for cap_area in sorted_areas[:2]:
        assert cap_area == pytest.approx(
            cylinder_geometry["expected_cap_area"], rel=0.05
        )
    # Largest is the curved lateral surface
    assert sorted_areas[2] == pytest.approx(
        cylinder_geometry["expected_lateral_area"], rel=0.05
    )
    assert sum(areas) == pytest.approx(
        cylinder_geometry["expected_total_surface_area"], rel=0.05
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_cell_id_rectangle(rectangle_geometry, backend):
    """Test surface areas of a 10x20x30 cuboid by cell ID.

    Planar faces are exact after triangulation, so we use a tight tolerance.
    """
    areas = di.get_surface_area_by_cell_id(
        filename=rectangle_geometry["filename"],
        cell_id=rectangle_geometry["cell_id"],
        backend=backend,
    )
    assert len(areas) == rectangle_geometry["expected_num_surfaces"]
    sorted_areas = sorted(areas)
    expected_sorted = rectangle_geometry["expected_sorted_areas"]
    for actual, expected in zip(sorted_areas, expected_sorted):
        assert actual == pytest.approx(expected, rel=1e-10)
    assert sum(areas) == pytest.approx(
        rectangle_geometry["expected_total_surface_area"], rel=1e-10
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_material_name_cube(cube_geometry, backend):
    """Test surface areas of a 10x10x10 cube by material name.

    Planar faces are exact after triangulation, so we use a tight tolerance.
    """
    areas = di.get_surface_area_by_material_name(
        filename=cube_geometry["filename"],
        material=cube_geometry["material"],
        backend=backend,
    )
    assert len(areas) == cube_geometry["expected_num_surfaces"]
    for area in sorted(areas):
        assert area == pytest.approx(
            cube_geometry["expected_surface_area_each"], rel=1e-10
        )
    assert sum(areas) == pytest.approx(
        cube_geometry["expected_total_surface_area"], rel=1e-10
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_material_name_sphere(sphere_geometry, backend):
    """Test surface area of a sphere with radius 5 by material name.

    Curved surfaces lose area to mesh discretization (~2%), so 5% tolerance.
    """
    areas = di.get_surface_area_by_material_name(
        filename=sphere_geometry["filename"],
        material=sphere_geometry["material"],
        backend=backend,
    )
    assert len(areas) >= 1
    assert sum(areas) == pytest.approx(
        sphere_geometry["expected_total_surface_area"], rel=0.05
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_material_name_cylinder(cylinder_geometry, backend):
    """Test surface areas of a cylinder (h=20, r=5) by material name.

    3 surfaces: two caps and one lateral. All have curved boundaries, so
    all use 5% tolerance.
    """
    areas = di.get_surface_area_by_material_name(
        filename=cylinder_geometry["filename"],
        material=cylinder_geometry["material"],
        backend=backend,
    )
    assert len(areas) == cylinder_geometry["expected_num_surfaces"]
    sorted_areas = sorted(areas)
    for cap_area in sorted_areas[:2]:
        assert cap_area == pytest.approx(
            cylinder_geometry["expected_cap_area"], rel=0.05
        )
    assert sorted_areas[2] == pytest.approx(
        cylinder_geometry["expected_lateral_area"], rel=0.05
    )
    assert sum(areas) == pytest.approx(
        cylinder_geometry["expected_total_surface_area"], rel=0.05
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_material_name_rectangle(rectangle_geometry, backend):
    """Test surface areas of a 10x20x30 cuboid by material name.

    Planar faces are exact after triangulation, so we use a tight tolerance.
    """
    areas = di.get_surface_area_by_material_name(
        filename=rectangle_geometry["filename"],
        material=rectangle_geometry["material"],
        backend=backend,
    )
    assert len(areas) == rectangle_geometry["expected_num_surfaces"]
    sorted_areas = sorted(areas)
    expected_sorted = rectangle_geometry["expected_sorted_areas"]
    for actual, expected in zip(sorted_areas, expected_sorted):
        assert actual == pytest.approx(expected, rel=1e-10)
    assert sum(areas) == pytest.approx(
        rectangle_geometry["expected_total_surface_area"], rel=1e-10
    )


# ============================================================================
# Tests for get_surface_area_by_surface_id
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_surface_id_cube(cube_geometry, backend):
    """A cube has 6 surfaces each with area 100."""
    result = di.get_surface_area_by_surface_id(
        filename=cube_geometry["filename"],
        backend=backend,
    )

    assert isinstance(result, dict)
    assert len(result) == cube_geometry["expected_num_surfaces"]
    for surf_id, area in result.items():
        assert isinstance(surf_id, int)
        assert area == pytest.approx(
            cube_geometry["expected_surface_area_each"], rel=0.05
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_surface_id_sphere(sphere_geometry, backend):
    """Total surface area of a sphere should match 4*pi*r^2."""
    result = di.get_surface_area_by_surface_id(
        filename=sphere_geometry["filename"],
        backend=backend,
    )

    assert len(result) == 1
    total_area = sum(result.values())
    assert total_area == pytest.approx(
        sphere_geometry["expected_total_surface_area"], rel=0.05
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_surface_id_cylinder(cylinder_geometry, backend):
    """A cylinder has 3 surfaces: two caps and one lateral."""
    result = di.get_surface_area_by_surface_id(
        filename=cylinder_geometry["filename"],
        backend=backend,
    )

    assert len(result) == cylinder_geometry["expected_num_surfaces"]
    total_area = sum(result.values())
    assert total_area == pytest.approx(
        cylinder_geometry["expected_total_surface_area"], rel=0.05
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_surface_id_keys_match_surfaces(cube_geometry, backend):
    """Surface IDs from get_surface_area_by_surface_id should match
    get_surface_ids."""
    areas = di.get_surface_area_by_surface_id(
        filename=cube_geometry["filename"],
        backend=backend,
    )
    surface_ids = di.get_surface_ids(
        filename=cube_geometry["filename"],
        backend=backend,
    )

    assert sorted(areas.keys()) == surface_ids


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_area_by_surface_id_file_not_found(backend):
    """Checks that a FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        di.get_surface_area_by_surface_id(
            filename="non_existant.h5m",
            backend=backend,
        )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_surface_area_by_surface_id_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return the same surface areas."""
    h5py_result = di.get_surface_area_by_surface_id(filename, backend="h5py")
    pymoab_result = di.get_surface_area_by_surface_id(filename, backend="pymoab")

    assert sorted(h5py_result.keys()) == sorted(pymoab_result.keys()), (
        f"Surface IDs differ: h5py={sorted(h5py_result.keys())}, "
        f"pymoab={sorted(pymoab_result.keys())}"
    )

    for surf_id in h5py_result:
        assert h5py_result[surf_id] == pytest.approx(
            pymoab_result[surf_id], rel=1e-10
        ), (
            f"Surface {surf_id} area differs: "
            f"h5py={h5py_result[surf_id]}, pymoab={pymoab_result[surf_id]}"
        )


# ============================================================================
# Tests for get_surface_ids_by_cell_id
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_ids_by_cell_id_cube(cube_geometry, backend):
    """A cube (cell_id=1) should have 6 bounding surfaces."""
    result = di.get_surface_ids_by_cell_id(
        filename=cube_geometry["filename"],
        cell_id=cube_geometry["cell_id"],
        backend=backend,
    )

    assert isinstance(result, list)
    assert result == sorted(result)
    assert len(result) == cube_geometry["expected_num_surfaces"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_ids_by_cell_id_touching_boxes(touching_boxes, backend):
    """Each volume in touching boxes should have bounding surfaces."""
    for cell_id in touching_boxes["volumes"]:
        result = di.get_surface_ids_by_cell_id(
            filename=touching_boxes["filename"],
            cell_id=cell_id,
            backend=backend,
        )
        assert len(result) > 0
        assert all(isinstance(s, int) for s in result)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_ids_by_cell_id_invalid(cube_geometry, backend):
    """Invalid cell_id should raise ValueError."""
    with pytest.raises(ValueError):
        di.get_surface_ids_by_cell_id(
            filename=cube_geometry["filename"],
            cell_id=999,
            backend=backend,
        )


# ============================================================================
# Tests for get_surface_ids_by_material_name
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_ids_by_material_name_cube(cube_geometry, backend):
    """A cube with material 'cube' should have 6 bounding surfaces."""
    result = di.get_surface_ids_by_material_name(
        filename=cube_geometry["filename"],
        material=cube_geometry["material"],
        backend=backend,
    )

    assert isinstance(result, list)
    assert result == sorted(result)
    assert len(result) == cube_geometry["expected_num_surfaces"]


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_ids_by_material_name_touching_boxes(touching_boxes, backend):
    """Each material in touching boxes should have bounding surfaces."""
    for material in touching_boxes["materials"]:
        result = di.get_surface_ids_by_material_name(
            filename=touching_boxes["filename"],
            material=material,
            backend=backend,
        )
        assert len(result) > 0


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_ids_by_material_name_invalid(cube_geometry, backend):
    """Invalid material name should raise ValueError."""
    with pytest.raises(ValueError):
        di.get_surface_ids_by_material_name(
            filename=cube_geometry["filename"],
            material="nonexistent",
            backend=backend,
        )


# ============================================================================
# Tests for get_surface_shared_status
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_shared_status_touching_boxes(touching_boxes, backend):
    """Touching boxes should have at least one shared surface with 2 cell_ids."""
    result = di.get_surface_shared_status(
        filename=touching_boxes["filename"],
        backend=backend,
    )

    # All entries should have the expected structure
    for surf_id, info in result.items():
        assert "materials" in info
        assert "cell_ids" in info
        assert len(info["materials"]) == len(info["cell_ids"])

    # At least one surface should be shared (2 cell_ids)
    shared = {k: v for k, v in result.items() if len(v["cell_ids"]) == 2}
    assert len(shared) >= 1

    # The shared surface should reference both materials
    for info in shared.values():
        assert set(info["materials"]) == {"small_box", "big_box"}
        assert set(info["cell_ids"]) == {1, 2}

    # Non-shared surfaces should have exactly 1 cell_id
    for surf_id, info in result.items():
        if surf_id not in shared:
            assert len(info["cell_ids"]) == 1


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_shared_status_separated_boxes(separated_boxes, backend):
    """Separated boxes should have no shared surfaces."""
    result = di.get_surface_shared_status(
        filename=separated_boxes["filename"],
        backend=backend,
    )

    for surf_id, info in result.items():
        assert len(info["cell_ids"]) <= 1


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_surface_shared_status_cube(cube_geometry, backend):
    """A single cube should have all surfaces with 1 cell_id."""
    result = di.get_surface_shared_status(
        filename=cube_geometry["filename"],
        backend=backend,
    )

    assert len(result) == cube_geometry["expected_num_surfaces"]
    for surf_id, info in result.items():
        assert info["cell_ids"] == [1]
        assert info["materials"] == ["cube"]


# ============================================================================
# Tests for rotate_around_axis
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_z(rectangle_geometry, tmp_path, backend):
    """Rotating a 10x20x30 rectangle 90 deg around z swaps x/y extents."""
    output = str(tmp_path / "rotated_z.h5m")
    di.rotate_around_axis(
        filename=rectangle_geometry["filename"],
        axis="z",
        degrees=90,
        output=output,
        backend=backend,
    )
    bbox = di.get_bounding_box(output)
    # Original: x in [-5, 5] (width 10), y in [-10, 10] (width 20)
    # After 90-deg z rotation: x-width should become ~20, y-width should become ~10
    assert bbox.width[0] == pytest.approx(20.0, rel=1e-6)
    assert bbox.width[1] == pytest.approx(10.0, rel=1e-6)
    assert bbox.width[2] == pytest.approx(30.0, rel=1e-6)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_x(rectangle_geometry, tmp_path, backend):
    """Rotating a 10x20x30 rectangle 90 deg around x swaps y/z extents."""
    output = str(tmp_path / "rotated_x.h5m")
    di.rotate_around_axis(
        filename=rectangle_geometry["filename"],
        axis="x",
        degrees=90,
        output=output,
        backend=backend,
    )
    bbox = di.get_bounding_box(output)
    # Original: y in [-10, 10] (width 20), z in [-15, 15] (width 30)
    # After 90-deg x rotation: y-width should become ~30, z-width should become ~20
    assert bbox.width[0] == pytest.approx(10.0, rel=1e-6)
    assert bbox.width[1] == pytest.approx(30.0, rel=1e-6)
    assert bbox.width[2] == pytest.approx(20.0, rel=1e-6)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_y(rectangle_geometry, tmp_path, backend):
    """Rotating a 10x20x30 rectangle 90 deg around y swaps x/z extents."""
    output = str(tmp_path / "rotated_y.h5m")
    di.rotate_around_axis(
        filename=rectangle_geometry["filename"],
        axis="y",
        degrees=90,
        output=output,
        backend=backend,
    )
    bbox = di.get_bounding_box(output)
    # Original: x in [-5, 5] (width 10), z in [-15, 15] (width 30)
    # After 90-deg y rotation: x-width should become ~30, z-width should become ~10
    assert bbox.width[0] == pytest.approx(30.0, rel=1e-6)
    assert bbox.width[1] == pytest.approx(20.0, rel=1e-6)
    assert bbox.width[2] == pytest.approx(10.0, rel=1e-6)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_360(rectangle_geometry, tmp_path, backend):
    """A 360-degree rotation should leave the bounding box unchanged."""
    output = str(tmp_path / "rotated_360.h5m")
    di.rotate_around_axis(
        filename=rectangle_geometry["filename"],
        axis="z",
        degrees=360,
        output=output,
        backend=backend,
    )
    original_bbox = di.get_bounding_box(rectangle_geometry["filename"])
    rotated_bbox = di.get_bounding_box(output)
    for i in range(3):
        assert rotated_bbox.width[i] == pytest.approx(original_bbox.width[i], rel=1e-6)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_materials_preserved(rectangle_geometry, tmp_path, backend):
    """Materials should be unchanged after rotation."""
    output = str(tmp_path / "rotated_mats.h5m")
    di.rotate_around_axis(
        filename=rectangle_geometry["filename"],
        axis="z",
        degrees=45,
        output=output,
        backend=backend,
    )
    original_mats = di.get_materials(rectangle_geometry["filename"])
    rotated_mats = di.get_materials(output)
    assert original_mats == rotated_mats


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_invalid_axis(rectangle_geometry, backend):
    """An invalid axis should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid axis"):
        di.rotate_around_axis(
            filename=rectangle_geometry["filename"],
            axis="a",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_file_not_found(backend):
    """A missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        di.rotate_around_axis(filename="nonexistent.h5m", backend=backend)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_rotate_around_axis_openmc_transport(touching_boxes, backend, tmp_path):
    """Verify that a rotated h5m file is a valid DAGMC geometry by running
    OpenMC fixed-source particle transport through it.
    """
    import openmc

    output = str(tmp_path / f"rotated_transport_{backend}.h5m")
    di.rotate_around_axis(
        filename=touching_boxes["filename"],
        axis="z",
        degrees=90,
        output=output,
        backend=backend,
    )

    # Set up cross sections (H1 only)
    xs_path = os.path.join(os.path.dirname(__file__), "ENDFB-7.1-NNDC_H1.h5")
    xs_xml = str(tmp_path / "cross_sections.xml")
    with open(xs_xml, "w") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<cross_sections>\n"
            f'  <library materials="H1" path="{xs_path}" type="neutron"/>\n'
            "</cross_sections>\n"
        )
    openmc.config["cross_sections"] = xs_xml

    # Create materials matching the DAGMC file
    vol_mat = di.get_volumes_and_materials(output)
    mat_names = sorted(set(vol_mat.values()))
    openmc_mats = []
    for name in mat_names:
        mat = openmc.Material(name=name)
        mat.add_nuclide("H1", 1.0, "ao")
        mat.set_density("g/cm3", 0.001)
        openmc_mats.append(mat)
    materials = openmc.Materials(openmc_mats)

    # DAGMC geometry
    dag_univ = openmc.DAGMCUniverse(filename=output)
    bound_dag_univ = dag_univ.bounded_universe()
    geometry = openmc.Geometry(root=bound_dag_univ)

    # Point source inside a known material volume
    first_mat = mat_names[0]
    bb = di.get_bounding_box(output, materials=first_mat)
    center = bb.center
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(center)
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14e6], [1])

    settings = openmc.Settings()
    settings.batches = 2
    settings.particles = 1000
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.source = source

    tally = openmc.Tally(name="flux")
    tally.scores = ["flux"]

    model = openmc.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=openmc.Tallies([tally]),
    )

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        output_file = model.run(output=False)
        sp = openmc.StatePoint(output_file)
        flux = sp.get_tally(name="flux").mean.flatten()[0]
        # Flux should be positive (particles traversed the geometry)
        assert flux > 0
    finally:
        os.chdir(original_dir)


# ============================================================================
# Tests for move
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_x(cube_geometry, tmp_path, backend):
    """Moving a cube by 100 in x shifts the bounding box center."""
    output = str(tmp_path / "moved_x.h5m")
    di.move(
        filename=cube_geometry["filename"],
        x=100.0,
        output=output,
        backend=backend,
    )
    original_bbox = di.get_bounding_box(cube_geometry["filename"])
    moved_bbox = di.get_bounding_box(output)
    assert moved_bbox.center[0] == pytest.approx(
        original_bbox.center[0] + 100.0, rel=1e-6
    )
    assert moved_bbox.center[1] == pytest.approx(original_bbox.center[1], rel=1e-6)
    assert moved_bbox.center[2] == pytest.approx(original_bbox.center[2], rel=1e-6)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_y(cube_geometry, tmp_path, backend):
    """Moving a cube by 50 in y shifts the bounding box center."""
    output = str(tmp_path / "moved_y.h5m")
    di.move(
        filename=cube_geometry["filename"],
        y=50.0,
        output=output,
        backend=backend,
    )
    original_bbox = di.get_bounding_box(cube_geometry["filename"])
    moved_bbox = di.get_bounding_box(output)
    assert moved_bbox.center[0] == pytest.approx(original_bbox.center[0], rel=1e-6)
    assert moved_bbox.center[1] == pytest.approx(
        original_bbox.center[1] + 50.0, rel=1e-6
    )
    assert moved_bbox.center[2] == pytest.approx(original_bbox.center[2], rel=1e-6)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_z(cube_geometry, tmp_path, backend):
    """Moving a cube by -30 in z shifts the bounding box center."""
    output = str(tmp_path / "moved_z.h5m")
    di.move(
        filename=cube_geometry["filename"],
        z=-30.0,
        output=output,
        backend=backend,
    )
    original_bbox = di.get_bounding_box(cube_geometry["filename"])
    moved_bbox = di.get_bounding_box(output)
    assert moved_bbox.center[0] == pytest.approx(original_bbox.center[0], rel=1e-6)
    assert moved_bbox.center[1] == pytest.approx(original_bbox.center[1], rel=1e-6)
    assert moved_bbox.center[2] == pytest.approx(
        original_bbox.center[2] - 30.0, rel=1e-6
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_xyz(cube_geometry, tmp_path, backend):
    """Moving a cube in all directions simultaneously."""
    output = str(tmp_path / "moved_xyz.h5m")
    di.move(
        filename=cube_geometry["filename"],
        x=10.0,
        y=20.0,
        z=30.0,
        output=output,
        backend=backend,
    )
    original_bbox = di.get_bounding_box(cube_geometry["filename"])
    moved_bbox = di.get_bounding_box(output)
    assert moved_bbox.center[0] == pytest.approx(
        original_bbox.center[0] + 10.0, rel=1e-6
    )
    assert moved_bbox.center[1] == pytest.approx(
        original_bbox.center[1] + 20.0, rel=1e-6
    )
    assert moved_bbox.center[2] == pytest.approx(
        original_bbox.center[2] + 30.0, rel=1e-6
    )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_zero(cube_geometry, tmp_path, backend):
    """Moving by (0, 0, 0) should leave the bounding box unchanged."""
    output = str(tmp_path / "moved_zero.h5m")
    di.move(
        filename=cube_geometry["filename"],
        x=0.0,
        y=0.0,
        z=0.0,
        output=output,
        backend=backend,
    )
    original_bbox = di.get_bounding_box(cube_geometry["filename"])
    moved_bbox = di.get_bounding_box(output)
    for i in range(3):
        assert moved_bbox.lower_left[i] == pytest.approx(
            original_bbox.lower_left[i], rel=1e-6
        )
        assert moved_bbox.upper_right[i] == pytest.approx(
            original_bbox.upper_right[i], rel=1e-6
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_materials_preserved(cube_geometry, tmp_path, backend):
    """Materials should be unchanged after moving."""
    output = str(tmp_path / "moved_mats.h5m")
    di.move(
        filename=cube_geometry["filename"],
        x=10.0,
        output=output,
        backend=backend,
    )
    original_mats = di.get_materials(cube_geometry["filename"])
    moved_mats = di.get_materials(output)
    assert original_mats == moved_mats


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_file_not_found(backend):
    """A missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        di.move(filename="nonexistent.h5m", backend=backend)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_move_openmc_transport(touching_boxes, backend, tmp_path):
    """Verify that a moved h5m file is a valid DAGMC geometry by running
    OpenMC fixed-source particle transport through it.
    """
    import openmc

    output = str(tmp_path / f"moved_transport_{backend}.h5m")
    di.move(
        filename=touching_boxes["filename"],
        x=100.0,
        y=50.0,
        z=-25.0,
        output=output,
        backend=backend,
    )

    # Set up cross sections (H1 only)
    xs_path = os.path.join(os.path.dirname(__file__), "ENDFB-7.1-NNDC_H1.h5")
    xs_xml = str(tmp_path / "cross_sections.xml")
    with open(xs_xml, "w") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<cross_sections>\n"
            f'  <library materials="H1" path="{xs_path}" type="neutron"/>\n'
            "</cross_sections>\n"
        )
    openmc.config["cross_sections"] = xs_xml

    # Create materials matching the DAGMC file
    vol_mat = di.get_volumes_and_materials(output)
    mat_names = sorted(set(vol_mat.values()))
    openmc_mats = []
    for name in mat_names:
        mat = openmc.Material(name=name)
        mat.add_nuclide("H1", 1.0, "ao")
        mat.set_density("g/cm3", 0.001)
        openmc_mats.append(mat)
    materials = openmc.Materials(openmc_mats)

    # DAGMC geometry
    dag_univ = openmc.DAGMCUniverse(filename=output)
    bound_dag_univ = dag_univ.bounded_universe()
    geometry = openmc.Geometry(root=bound_dag_univ)

    # Point source inside a known material volume
    first_mat = mat_names[0]
    bb = di.get_bounding_box(output, materials=first_mat)
    center = bb.center
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(center)
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14e6], [1])

    settings = openmc.Settings()
    settings.batches = 2
    settings.particles = 1000
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.source = source

    tally = openmc.Tally(name="flux")
    tally.scores = ["flux"]

    model = openmc.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=openmc.Tallies([tally]),
    )

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        output_file = model.run(output=False)
        sp = openmc.StatePoint(output_file)
        flux = sp.get_tally(name="flux").mean.flatten()[0]
        # Flux should be positive (particles traversed the geometry)
        assert flux > 0
    finally:
        os.chdir(original_dir)


# ============================================================================
# Tests for combine_h5m_files
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_combine_two_separate_cubes(tmp_path, backend):
    """Combine two separate single-cube h5m files and verify volumes, materials,
    and bounding box of the result."""
    import cadquery as cq
    from cad_to_dagmc import CadToDagmc

    # Create cube A at origin
    cube_a = cq.Workplane().box(10, 10, 10)
    file_a = str(tmp_path / "cube_a.h5m")
    model_a = CadToDagmc()
    model_a.add_cadquery_object(cadquery_object=cube_a, material_tags=["mat_a"])
    model_a.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=file_a
    )

    # Create cube B offset in x (no overlap)
    cube_b = cq.Workplane().moveTo(30, 0).box(10, 10, 10)
    file_b = str(tmp_path / "cube_b.h5m")
    model_b = CadToDagmc()
    model_b.add_cadquery_object(cadquery_object=cube_b, material_tags=["mat_b"])
    model_b.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=file_b
    )

    output = str(tmp_path / "combined.h5m")
    di.combine_h5m_files(
        input_files=[file_a, file_b], output_file=output, backend=backend
    )

    # Check volumes
    volumes = di.get_volumes(output)
    assert volumes == [1, 2]

    # Check materials
    materials = di.get_materials(output)
    assert sorted(materials) == ["mat_a", "mat_b"]

    # Check volume-material mapping
    vol_mat = di.get_volumes_and_materials(output, remove_prefix=True)
    assert vol_mat == {1: "mat_a", 2: "mat_b"}

    # Combined bounding box should span both cubes
    bbox = di.get_bounding_box(output)
    np.testing.assert_allclose(bbox.lower_left, [-5.0, -5.0, -5.0], atol=0.1)
    np.testing.assert_allclose(bbox.upper_right, [35.0, 5.0, 5.0], atol=0.1)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_combine_preserves_volumes(cube_geometry, sphere_geometry, tmp_path, backend):
    """Combining a cube and sphere preserves the mesh volume of each."""
    output = str(tmp_path / "combined_vol.h5m")
    di.combine_h5m_files(
        input_files=[cube_geometry["filename"], sphere_geometry["filename"]],
        output_file=output,
        backend=backend,
    )

    vol_by_id = di.get_volumes_by_cell_id(output)
    # Volume 1 is the cube (1000), volume 2 is the sphere (4/3 * pi * 125)
    assert vol_by_id[1] == pytest.approx(1000.0, rel=0.05)
    assert vol_by_id[2] == pytest.approx(4 / 3 * np.pi * 125, rel=0.05)


def test_combine_empty_list():
    """Passing an empty list raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        di.combine_h5m_files(input_files=[], output_file="out.h5m")


def test_combine_file_not_found():
    """Passing a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        di.combine_h5m_files(input_files=["does_not_exist.h5m"], output_file="out.h5m")


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_combine_openmc_transport(tmp_path, backend):
    """Verify that a combined h5m file is a valid DAGMC geometry by running
    OpenMC fixed-source particle transport through it."""
    import cadquery as cq
    import openmc
    from cad_to_dagmc import CadToDagmc

    # Create cube A at origin
    cube_a = cq.Workplane().box(10, 10, 10)
    file_a = str(tmp_path / "cube_a.h5m")
    model_a = CadToDagmc()
    model_a.add_cadquery_object(cadquery_object=cube_a, material_tags=["mat_a"])
    model_a.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=file_a
    )

    # Create cube B offset in x (no overlap)
    cube_b = cq.Workplane().moveTo(30, 0).box(10, 10, 10)
    file_b = str(tmp_path / "cube_b.h5m")
    model_b = CadToDagmc()
    model_b.add_cadquery_object(cadquery_object=cube_b, material_tags=["mat_b"])
    model_b.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=file_b
    )

    output = str(tmp_path / "combined_transport.h5m")
    di.combine_h5m_files(
        input_files=[file_a, file_b], output_file=output, backend=backend
    )

    # Set up cross sections (H1 only)
    xs_path = os.path.join(os.path.dirname(__file__), "ENDFB-7.1-NNDC_H1.h5")
    xs_xml = str(tmp_path / "cross_sections.xml")
    with open(xs_xml, "w") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<cross_sections>\n"
            f'  <library materials="H1" path="{xs_path}" type="neutron"/>\n'
            "</cross_sections>\n"
        )
    openmc.config["cross_sections"] = xs_xml

    # Create materials matching the DAGMC file
    vol_mat = di.get_volumes_and_materials(output)
    mat_names = sorted(set(vol_mat.values()))
    openmc_mats = []
    for name in mat_names:
        mat = openmc.Material(name=name)
        mat.add_nuclide("H1", 1.0, "ao")
        mat.set_density("g/cm3", 0.001)
        openmc_mats.append(mat)
    materials = openmc.Materials(openmc_mats)

    # DAGMC geometry
    dag_univ = openmc.DAGMCUniverse(filename=output)
    bound_dag_univ = dag_univ.bounded_universe()
    geometry = openmc.Geometry(root=bound_dag_univ)

    # Point source near center of geometry
    bb = di.get_bounding_box(output)
    center = bb.center
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(
        (center[0] + 0.1, center[1] + 0.1, center[2] + 0.1)
    )
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14e6], [1])

    settings = openmc.Settings()
    settings.batches = 2
    settings.particles = 1000
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.source = source

    tally = openmc.Tally(name="flux")
    tally.scores = ["flux"]

    model = openmc.Model(
        materials=materials,
        geometry=geometry,
        settings=settings,
        tallies=openmc.Tallies([tally]),
    )

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        output_file = model.run(output=False)
        sp = openmc.StatePoint(output_file)
        flux = sp.get_tally(name="flux").mean.flatten()[0]
        # Flux should be positive (particles traversed the geometry)
        assert flux > 0
    finally:
        os.chdir(original_dir)


# ============================================================================
# Tests for set_boundary_condition
# ============================================================================


def test_set_boundary_condition_vacuum(touching_boxes, tmp_path):
    """Setting a vacuum boundary condition creates a Group with the correct NAME."""
    output = str(tmp_path / "bc_vacuum.h5m")
    surface_ids = di.get_surface_ids(touching_boxes["filename"])
    target_id = surface_ids[0]

    result = di.set_boundary_condition(
        touching_boxes["filename"], target_id, "vacuum", output
    )
    assert result == output

    # Verify the boundary group was written
    with h5py.File(output, "r") as f:
        name_ids = f["tstt/tags/NAME/id_list"][()]
        name_vals = f["tstt/tags/NAME/values"][()]
        names = {}
        for h, v in zip(name_ids, name_vals):
            data = v.tobytes() if hasattr(v, "tobytes") else bytes(v)
            names[int(h)] = data.split(b"\x00", 1)[0].decode("ascii")
        assert "boundary:vacuum" in names.values()

    # Geometry should still be intact
    assert di.get_surface_ids(output) == surface_ids
    assert di.get_volumes(output) == di.get_volumes(touching_boxes["filename"])


def test_set_boundary_condition_reflective(touching_boxes, tmp_path):
    """Setting a reflective boundary condition also works."""
    output = str(tmp_path / "bc_reflect.h5m")
    surface_ids = di.get_surface_ids(touching_boxes["filename"])

    di.set_boundary_condition(
        touching_boxes["filename"], surface_ids[-1], "reflective", output
    )

    with h5py.File(output, "r") as f:
        name_ids = f["tstt/tags/NAME/id_list"][()]
        name_vals = f["tstt/tags/NAME/values"][()]
        names = []
        for h, v in zip(name_ids, name_vals):
            data = v.tobytes() if hasattr(v, "tobytes") else bytes(v)
            names.append(data.split(b"\x00", 1)[0].decode("ascii"))
        assert "boundary:reflective" in names


def test_set_boundary_condition_invalid_surface(touching_boxes, tmp_path):
    """Requesting a non-existent surface ID raises ValueError."""
    output = str(tmp_path / "bc_bad.h5m")
    with pytest.raises(ValueError, match="not found"):
        di.set_boundary_condition(touching_boxes["filename"], 9999, "vacuum", output)


def test_set_boundary_condition_file_not_found():
    """A missing input file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        di.set_boundary_condition("nonexistent.h5m", 1, "vacuum", "/tmp/out.h5m")


def test_set_boundary_condition_in_place(touching_boxes, tmp_path):
    """Passing output_filename=None modifies the file in place."""
    import shutil

    copy_path = str(tmp_path / "inplace.h5m")
    shutil.copy2(touching_boxes["filename"], copy_path)

    surface_ids = di.get_surface_ids(copy_path)
    di.set_boundary_condition(copy_path, surface_ids[0], "vacuum")

    with h5py.File(copy_path, "r") as f:
        name_ids = f["tstt/tags/NAME/id_list"][()]
        name_vals = f["tstt/tags/NAME/values"][()]
        names = []
        for h, v in zip(name_ids, name_vals):
            data = v.tobytes() if hasattr(v, "tobytes") else bytes(v)
            names.append(data.split(b"\x00", 1)[0].decode("ascii"))
        assert "boundary:vacuum" in names
