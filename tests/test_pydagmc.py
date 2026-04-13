import warnings

import numpy as np
import pydagmc
import pytest

import dagmc_h5m_file_inspector as di

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
def test_volume_sizes_pydagmc_consistency(filename):
    """Verify our volume calculations match pydagmc results"""

    # Get volumes from our implementations
    h5py_volumes = di.get_volumes_by_cell_id(filename, backend="h5py")

    # Get volumes from pydagmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag_model = pydagmc.Model(filename)

    pydagmc_volumes = {
        int(vol_id): float(vol.volume)
        for vol_id, vol in dag_model.volumes_by_id.items()
    }

    # Check same volume IDs are returned
    assert set(h5py_volumes.keys()) == set(pydagmc_volumes.keys()), (
        f"Volume IDs differ: h5py={set(h5py_volumes.keys())}, "
        f"pydagmc={set(pydagmc_volumes.keys())}"
    )

    # Check volumes match within tolerance
    for vol_id in h5py_volumes:
        h5py_vol = h5py_volumes[vol_id]
        pydagmc_vol = pydagmc_volumes[vol_id]

        # Use relative tolerance for non-zero volumes
        if pydagmc_vol > 1e-10:
            rel_diff = abs(h5py_vol - pydagmc_vol) / pydagmc_vol
            assert rel_diff < 0.01, (
                f"Volume {vol_id} differs: h5py={h5py_vol}, "
                f"pydagmc={pydagmc_vol}, rel_diff={rel_diff}"
            )
        else:
            # For near-zero volumes, use absolute tolerance
            assert abs(h5py_vol - pydagmc_vol) < 1e-6, (
                f"Volume {vol_id} differs: h5py={h5py_vol}, pydagmc={pydagmc_vol}"
            )


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
def test_triangle_conn_and_coords_pydagmc_consistency(filename):
    """Verify our triangle data matches pydagmc results"""

    # Get data from our implementation
    h5py_data = di.get_triangle_conn_and_coords_by_volume(filename, backend="h5py")

    # Get data from pydagmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag_model = pydagmc.Model(filename)

    for vol_id, volume in dag_model.volumes_by_id.items():
        vol_id = int(vol_id)

        # Get pydagmc triangle data (with compress=True for unique vertices)
        pydagmc_conn, pydagmc_coords = volume.get_triangle_conn_and_coords(
            compress=True
        )

        # Get our data
        assert vol_id in h5py_data, f"Volume {vol_id} missing from our implementation"
        our_conn, our_coords = h5py_data[vol_id]

        # Same number of triangles
        assert len(our_conn) == len(pydagmc_conn), (
            f"Volume {vol_id}: triangle count differs - "
            f"ours={len(our_conn)}, pydagmc={len(pydagmc_conn)}"
        )

        # Same number of unique vertices
        assert len(our_coords) == len(pydagmc_coords), (
            f"Volume {vol_id}: vertex count differs - "
            f"ours={len(our_coords)}, pydagmc={len(pydagmc_coords)}"
        )

        # Verify the actual geometry is equivalent by checking:
        # 1. The set of unique coordinates should match
        # 2. The triangles should form the same mesh

        # Sort coordinates and compare
        our_sorted = np.sort(our_coords, axis=0)
        pydagmc_sorted = np.sort(pydagmc_coords, axis=0)
        np.testing.assert_allclose(
            our_sorted,
            pydagmc_sorted,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Volume {vol_id}: coordinates differ from pydagmc",
        )

        # Verify triangles reference valid vertices and form same geometry
        # by comparing the actual 3D coordinates of each triangle's vertices
        our_tri_coords = our_coords[our_conn]  # Shape: (n_triangles, 3, 3)
        pydagmc_tri_coords = pydagmc_coords[pydagmc_conn]

        # Sort triangles for comparison (triangles may be in different order)
        # Sort each triangle's vertices, then sort all triangles
        our_tri_sorted = np.sort(our_tri_coords.reshape(-1, 9), axis=1)
        our_tri_sorted = our_tri_sorted[np.lexsort(our_tri_sorted.T)]

        pydagmc_tri_sorted = np.sort(pydagmc_tri_coords.reshape(-1, 9), axis=1)
        pydagmc_tri_sorted = pydagmc_tri_sorted[np.lexsort(pydagmc_tri_sorted.T)]

        np.testing.assert_allclose(
            our_tri_sorted,
            pydagmc_tri_sorted,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Volume {vol_id}: triangle geometry differs from pydagmc",
        )
