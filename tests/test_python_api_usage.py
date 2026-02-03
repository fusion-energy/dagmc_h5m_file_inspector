import pytest
import numpy as np
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


@pytest.mark.parametrize("backend", [
    pytest.param("h5py", marks=pytest.mark.xfail(reason="h5py volume size calculation not yet implemented")),
    "pymoab",
])
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


@pytest.mark.parametrize("backend", [
    pytest.param("h5py", marks=pytest.mark.xfail(reason="h5py volume size calculation not yet implemented")),
    "pymoab",
])
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
