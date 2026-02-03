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
def test_volume_sizes(touching_boxes, backend):
    """Extracts the geometric volumes from a dagmc file and checks they
    match the expected cube volumes"""

    volume_sizes = di.get_volumes_sizes_from_h5m(
        filename=touching_boxes['filename'],
        backend=backend,
    )

    expected = touching_boxes['expected_volume_sizes']

    for vol_id, expected_size in expected.items():
        assert vol_id in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[vol_id] - expected_size) / expected_size < 0.05


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
def test_separated_volume_sizes(separated_boxes, backend):
    """Extracts the geometric volumes from separated boxes and checks they
    match the expected cube volumes"""

    volume_sizes = di.get_volumes_sizes_from_h5m(
        filename=separated_boxes['filename'],
        backend=backend,
    )

    expected = separated_boxes['expected_volume_sizes']

    for vol_id, expected_size in expected.items():
        assert vol_id in volume_sizes
        # Allow 5% tolerance for mesh discretization
        assert abs(volume_sizes[vol_id] - expected_size) / expected_size < 0.05


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
