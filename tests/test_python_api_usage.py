import os

import pytest
import numpy as np
import openmc
import dagmc_h5m_file_inspector as di


# ============================================================================
# Tests for touching boxes geometry
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_and_material_extraction_without_stripped_prefix(touching_boxes, backend):
    """Extracts the volume numbers and material ids from a dagmc file and
    checks the contents match the expected contents"""

    dict_of_vol_and_mats = di.get_volumes_and_materials_from_h5m(
        filename=touching_boxes['filename'],
        remove_prefix=False,
        backend=backend,
    )

    assert dict_of_vol_and_mats == touching_boxes['volumes_and_materials_with_prefix']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_and_material_extraction_remove_prefix(touching_boxes, backend):
    """Extracts the volume numbers and material ids from a dagmc file and
    checks the contents match the expected contents"""

    dict_of_vol_and_mats = di.get_volumes_and_materials_from_h5m(
        filename=touching_boxes['filename'],
        backend=backend,
    )

    assert dict_of_vol_and_mats == touching_boxes['volumes_and_materials']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_extraction(touching_boxes, backend):
    """Extracts the volume ids from a dagmc file and checks the contents
    match the expected contents"""

    volumes = di.get_volumes_from_h5m(
        filename=touching_boxes['filename'],
        backend=backend,
    )

    assert volumes == touching_boxes['volumes']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_material_extraction_no_remove_prefix(touching_boxes, backend):
    """Extracts the materials tags from a dagmc file and checks the
    contents match the expected contents"""

    materials = di.get_materials_from_h5m(
        filename=touching_boxes['filename'],
        remove_prefix=False,
        backend=backend,
    )

    assert materials == touching_boxes['materials_with_prefix']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_material_extraction_remove_prefix(touching_boxes, backend):
    """Extracts the materials tags from a dagmc file and checks the
    contents match the expected contents"""

    materials = di.get_materials_from_h5m(
        filename=touching_boxes['filename'],
        backend=backend,
    )

    assert materials == touching_boxes['materials']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_fail_with_missing_input_files(backend):
    """Calls functions without necessary input files to check if error
    handling is working"""

    with pytest.raises(FileNotFoundError):
        di.get_volumes_and_materials_from_h5m(
            filename="non_existant.h5m",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_bounding_box(touching_boxes, backend):
    """Extracts the bounding box from a dagmc file and checks it matches
    the expected geometry bounds"""

    lower_left, upper_right = di.get_bounding_box_from_h5m(
        filename=touching_boxes['filename'],
        backend=backend,
    )

    np.testing.assert_allclose(lower_left, touching_boxes['lower_left'], rtol=1e-5)
    np.testing.assert_allclose(upper_right, touching_boxes['upper_right'], rtol=1e-5)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_sizes_by_cell_id(touching_boxes, backend):
    """Extracts the geometric volumes from a dagmc file and checks they
    match the expected cube volumes"""

    volume_sizes = di.get_volumes_sizes_from_h5m_by_cell_id(
        filename=touching_boxes['filename'],
        backend=backend,
    )

    expected = touching_boxes['expected_volume_sizes']

    for vol_id, expected_size in expected.items():
        assert vol_id in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[vol_id] - expected_size) / expected_size < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_sizes_by_material_name(touching_boxes, backend):
    """Extracts the geometric volumes by material name from a dagmc file"""

    volume_sizes = di.get_volumes_sizes_from_h5m_by_material_name(
        filename=touching_boxes['filename'],
        backend=backend,
    )

    # small_box is volume 1 (1000), big_box is volume 2 (8000)
    expected = {
        'small_box': touching_boxes['expected_volume_sizes'][1],
        'big_box': touching_boxes['expected_volume_sizes'][2],
    }

    for mat_name, expected_size in expected.items():
        assert mat_name in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[mat_name] - expected_size) / expected_size < 0.05


# ============================================================================
# Tests for separated boxes geometry
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_and_material_extraction(separated_boxes, backend):
    """Extracts the volume numbers and material ids from separated boxes"""

    dict_of_vol_and_mats = di.get_volumes_and_materials_from_h5m(
        filename=separated_boxes['filename'],
        backend=backend,
    )

    assert dict_of_vol_and_mats == separated_boxes['volumes_and_materials']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_extraction(separated_boxes, backend):
    """Extracts the volume ids from separated boxes"""

    volumes = di.get_volumes_from_h5m(
        filename=separated_boxes['filename'],
        backend=backend,
    )

    assert volumes == separated_boxes['volumes']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_material_extraction(separated_boxes, backend):
    """Extracts the materials tags from separated boxes"""

    materials = di.get_materials_from_h5m(
        filename=separated_boxes['filename'],
        backend=backend,
    )

    assert materials == separated_boxes['materials']


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_bounding_box(separated_boxes, backend):
    """Extracts the bounding box from separated boxes"""

    lower_left, upper_right = di.get_bounding_box_from_h5m(
        filename=separated_boxes['filename'],
        backend=backend,
    )

    np.testing.assert_allclose(lower_left, separated_boxes['lower_left'], rtol=1e-5)
    np.testing.assert_allclose(upper_right, separated_boxes['upper_right'], rtol=1e-5)


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_sizes_by_cell_id(separated_boxes, backend):
    """Extracts the geometric volumes from separated boxes and checks they
    match the expected cube volumes"""

    volume_sizes = di.get_volumes_sizes_from_h5m_by_cell_id(
        filename=separated_boxes['filename'],
        backend=backend,
    )

    expected = separated_boxes['expected_volume_sizes']

    for vol_id, expected_size in expected.items():
        assert vol_id in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[vol_id] - expected_size) / expected_size < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_separated_volume_sizes_by_material_name(separated_boxes, backend):
    """Extracts the geometric volumes by material name from separated boxes"""

    volume_sizes = di.get_volumes_sizes_from_h5m_by_material_name(
        filename=separated_boxes['filename'],
        backend=backend,
    )

    # box_a is volume 1, box_b is volume 2
    expected = {
        'box_a': separated_boxes['expected_volume_sizes'][1],
        'box_b': separated_boxes['expected_volume_sizes'][2],
    }

    for mat_name, expected_size in expected.items():
        assert mat_name in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[mat_name] - expected_size) / expected_size < 0.05


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

    h5py_volumes = di.get_volumes_from_h5m(filename, backend="h5py")
    pymoab_volumes = di.get_volumes_from_h5m(filename, backend="pymoab")

    assert h5py_volumes == pymoab_volumes, \
        f"Volume IDs differ: h5py={h5py_volumes}, pymoab={pymoab_volumes}"


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_material_tags_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return the same material tags"""

    h5py_materials = di.get_materials_from_h5m(filename, backend="h5py")
    pymoab_materials = di.get_materials_from_h5m(filename, backend="pymoab")

    assert h5py_materials == pymoab_materials, \
        f"Material tags differ: h5py={h5py_materials}, pymoab={pymoab_materials}"


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volumes_and_materials_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends return the same volume-to-material mapping"""

    h5py_mapping = di.get_volumes_and_materials_from_h5m(filename, backend="h5py")
    pymoab_mapping = di.get_volumes_and_materials_from_h5m(filename, backend="pymoab")

    assert h5py_mapping == pymoab_mapping, \
        f"Volume-material mapping differs: h5py={h5py_mapping}, pymoab={pymoab_mapping}"


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volume_sizes_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends produce the same volume calculations"""

    h5py_volumes = di.get_volumes_sizes_from_h5m_by_cell_id(filename, backend="h5py")
    pymoab_volumes = di.get_volumes_sizes_from_h5m_by_cell_id(filename, backend="pymoab")

    # Check same volume IDs are returned
    assert set(h5py_volumes.keys()) == set(pymoab_volumes.keys()), \
        f"Volume IDs differ: h5py={set(h5py_volumes.keys())}, pymoab={set(pymoab_volumes.keys())}"

    # Check volumes match within tolerance
    for vol_id in h5py_volumes:
        h5py_vol = h5py_volumes[vol_id]
        pymoab_vol = pymoab_volumes[vol_id]

        # Use relative tolerance for non-zero volumes
        if pymoab_vol > 1e-10:
            rel_diff = abs(h5py_vol - pymoab_vol) / pymoab_vol
            assert rel_diff < 0.01, \
                f"Volume {vol_id} differs: h5py={h5py_vol}, pymoab={pymoab_vol}, rel_diff={rel_diff}"
        else:
            # For near-zero volumes, use absolute tolerance
            assert abs(h5py_vol - pymoab_vol) < 1e-6, \
                f"Volume {vol_id} differs: h5py={h5py_vol}, pymoab={pymoab_vol}"


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volume_sizes_by_material_h5py_pymoab_consistency(filename):
    """Verify h5py and pymoab backends produce the same volume calculations by material"""

    h5py_volumes = di.get_volumes_sizes_from_h5m_by_material_name(filename, backend="h5py")
    pymoab_volumes = di.get_volumes_sizes_from_h5m_by_material_name(filename, backend="pymoab")

    # Check same material names are returned
    assert set(h5py_volumes.keys()) == set(pymoab_volumes.keys()), \
        f"Material names differ: h5py={set(h5py_volumes.keys())}, pymoab={set(pymoab_volumes.keys())}"

    # Check volumes match within tolerance
    for mat_name in h5py_volumes:
        h5py_vol = h5py_volumes[mat_name]
        pymoab_vol = pymoab_volumes[mat_name]

        # Use relative tolerance for non-zero volumes
        if pymoab_vol > 1e-10:
            rel_diff = abs(h5py_vol - pymoab_vol) / pymoab_vol
            assert rel_diff < 0.01, \
                f"Material '{mat_name}' differs: h5py={h5py_vol}, pymoab={pymoab_vol}, rel_diff={rel_diff}"
        else:
            # For near-zero volumes, use absolute tolerance
            assert abs(h5py_vol - pymoab_vol) < 1e-6, \
                f"Material '{mat_name}' differs: h5py={h5py_vol}, pymoab={pymoab_vol}"


# ============================================================================
# Tests for OpenMC material volume assignment
# ============================================================================


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_with_list(touching_boxes, backend):
    """Tests setting volumes on a list of OpenMC materials"""

    # Create OpenMC materials matching the DAGMC material names
    small_box_mat = openmc.Material(name='small_box')
    big_box_mat = openmc.Material(name='big_box')
    materials = [small_box_mat, big_box_mat]

    # Initially volumes should be None
    assert small_box_mat.volume is None
    assert big_box_mat.volume is None

    # Set volumes from DAGMC file
    di.set_openmc_material_volumes_from_h5m(
        materials=materials,
        filename=touching_boxes['filename'],
        backend=backend,
    )

    # Check volumes are set correctly (with 5% tolerance for mesh discretization)
    expected = touching_boxes['expected_volume_sizes']
    # small_box is volume 1, big_box is volume 2
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05
    assert abs(big_box_mat.volume - expected[2]) / expected[2] < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_with_materials_object(touching_boxes, backend):
    """Tests setting volumes on an OpenMC Materials collection"""

    # Create OpenMC materials and add to Materials collection
    small_box_mat = openmc.Material(name='small_box')
    big_box_mat = openmc.Material(name='big_box')
    materials = openmc.Materials([small_box_mat, big_box_mat])

    # Set volumes from DAGMC file
    di.set_openmc_material_volumes_from_h5m(
        materials=materials,
        filename=touching_boxes['filename'],
        backend=backend,
    )

    # Check volumes are set correctly
    expected = touching_boxes['expected_volume_sizes']
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05
    assert abs(big_box_mat.volume - expected[2]) / expected[2] < 0.05


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_non_matching_materials(touching_boxes, backend):
    """Tests that materials without matching names are not affected"""

    # Create materials - one matching, one not
    small_box_mat = openmc.Material(name='small_box')
    unmatched_mat = openmc.Material(name='nonexistent_material')
    materials = [small_box_mat, unmatched_mat]

    # Set volumes from DAGMC file
    di.set_openmc_material_volumes_from_h5m(
        materials=materials,
        filename=touching_boxes['filename'],
        backend=backend,
    )

    # Matching material should have volume set
    expected = touching_boxes['expected_volume_sizes']
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05

    # Non-matching material should remain None
    assert unmatched_mat.volume is None


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_duplicate_names_error(touching_boxes, backend):
    """Tests that duplicate material names raise an error"""

    # Create materials with duplicate names
    mat1 = openmc.Material(name='small_box')
    mat2 = openmc.Material(name='small_box')  # Duplicate!
    materials = [mat1, mat2]

    # Should raise ValueError for duplicate names
    with pytest.raises(ValueError, match="Multiple OpenMC materials have the same name"):
        di.set_openmc_material_volumes_from_h5m(
            materials=materials,
            filename=touching_boxes['filename'],
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_file_not_found(backend):
    """Tests that missing file raises FileNotFoundError"""

    mat = openmc.Material(name='test')
    materials = [mat]

    with pytest.raises(FileNotFoundError):
        di.set_openmc_material_volumes_from_h5m(
            materials=materials,
            filename='nonexistent_file.h5m',
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_set_openmc_material_volumes_with_none_names(touching_boxes, backend):
    """Tests that materials with None names are ignored without error"""

    # Create materials - one with name, one without
    small_box_mat = openmc.Material(name='small_box')
    unnamed_mat = openmc.Material()  # No name (defaults to None)
    materials = [small_box_mat, unnamed_mat]

    # Should not raise error, unnamed materials are skipped
    di.set_openmc_material_volumes_from_h5m(
        materials=materials,
        filename=touching_boxes['filename'],
        backend=backend,
    )

    # Named material should have volume set
    expected = touching_boxes['expected_volume_sizes']
    assert abs(small_box_mat.volume - expected[1]) / expected[1] < 0.05

    # Unnamed material should remain unchanged
    assert unnamed_mat.volume is None


# ============================================================================
# Tests comparing volume calculations with pydagmc and OpenMC stochastic
# ============================================================================


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_volume_sizes_pydagmc_consistency(filename):
    """Verify our volume calculations match pydagmc results"""
    import warnings
    import pydagmc

    # Get volumes from our implementations
    h5py_volumes = di.get_volumes_sizes_from_h5m_by_cell_id(filename, backend="h5py")

    # Get volumes from pydagmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag_model = pydagmc.Model(filename)

    pydagmc_volumes = {
        int(vol_id): float(vol.volume)
        for vol_id, vol in dag_model.volumes_by_id.items()
    }

    # Check same volume IDs are returned
    assert set(h5py_volumes.keys()) == set(pydagmc_volumes.keys()), \
        f"Volume IDs differ: h5py={set(h5py_volumes.keys())}, pydagmc={set(pydagmc_volumes.keys())}"

    # Check volumes match within tolerance
    for vol_id in h5py_volumes:
        h5py_vol = h5py_volumes[vol_id]
        pydagmc_vol = pydagmc_volumes[vol_id]

        # Use relative tolerance for non-zero volumes
        if pydagmc_vol > 1e-10:
            rel_diff = abs(h5py_vol - pydagmc_vol) / pydagmc_vol
            assert rel_diff < 0.01, \
                f"Volume {vol_id} differs: h5py={h5py_vol}, pydagmc={pydagmc_vol}, rel_diff={rel_diff}"
        else:
            # For near-zero volumes, use absolute tolerance
            assert abs(h5py_vol - pydagmc_vol) < 1e-6, \
                f"Volume {vol_id} differs: h5py={h5py_vol}, pydagmc={pydagmc_vol}"


# Subset of files for OpenMC stochastic tests (faster execution)
H5M_TEST_FILES_OPENMC_STOCHASTIC = [
    "tests/cuboid.h5m",
    "tests/sphere.h5m",
    "tests/nestedsphere.h5m",
    "tests/cylinder.h5m",
]


@pytest.mark.parametrize("filename", H5M_TEST_FILES_OPENMC_STOCHASTIC)
@pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="OpenMC stochastic volume tests skipped in CI (requires cross sections data)"
)
def test_volume_sizes_openmc_stochastic_consistency(filename, tmp_path):
    """Verify our volume calculations match OpenMC stochastic results.

    OpenMC uses Monte Carlo sampling to estimate volumes, so we allow
    a larger tolerance (5%) to account for statistical noise.

    This test is skipped in CI environments as it requires OpenMC cross
    sections data to be installed.
    """
    from pathlib import Path

    # Convert to absolute path before changing directories
    abs_filename = str(Path(filename).resolve())

    # Get volumes and materials from our implementation
    h5py_volumes = di.get_volumes_sizes_from_h5m_by_cell_id(abs_filename, backend="h5py")
    lower, upper = di.get_bounding_box_from_h5m(abs_filename)
    materials_list = di.get_materials_from_h5m(abs_filename, remove_prefix=True)

    # Create OpenMC materials matching the DAGMC file
    openmc_mats = []
    for mat_name in materials_list:
        mat = openmc.Material(name=mat_name)
        mat.add_nuclide('H1', 1.0)
        mat.set_density('g/cm3', 1.0)
        openmc_mats.append(mat)
    materials = openmc.Materials(openmc_mats)

    # Create DAGMC universe and geometry
    dagmc_univ = openmc.DAGMCUniverse(abs_filename, auto_geom_ids=True)
    bounded_univ = dagmc_univ.bounded_universe()
    geometry = openmc.Geometry(bounded_univ)

    # Settings for volume calculation
    settings = openmc.Settings()
    settings.run_mode = 'volume'

    # Create volume calculation for all materials
    vol_calc = openmc.VolumeCalculation(
        domains=openmc_mats,
        samples=50000,
        lower_left=lower.tolist(),
        upper_right=upper.tolist()
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
        results = openmc.VolumeCalculation.from_hdf5('volume_1.h5')

        # Get OpenMC volumes by material name
        openmc_volumes_by_mat = {}
        for domain, vol in results.volumes.items():
            # domain is the material ID
            for mat in openmc_mats:
                if mat.id == domain:
                    openmc_volumes_by_mat[mat.name] = vol.nominal_value
                    break

        # Get our volumes by material name for comparison
        h5py_volumes_by_mat = di.get_volumes_sizes_from_h5m_by_material_name(abs_filename, backend="h5py")

        # Compare volumes (allow 5% tolerance for stochastic noise)
        for mat_name in h5py_volumes_by_mat:
            if mat_name in openmc_volumes_by_mat:
                h5py_vol = h5py_volumes_by_mat[mat_name]
                openmc_vol = openmc_volumes_by_mat[mat_name]

                if openmc_vol > 1e-10:
                    rel_diff = abs(h5py_vol - openmc_vol) / openmc_vol
                    assert rel_diff < 0.05, \
                        f"Material '{mat_name}' volume differs: h5py={h5py_vol}, openmc={openmc_vol}, rel_diff={rel_diff}"
    finally:
        os.chdir(original_dir)
