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
@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_volume_sizes_pydagmc_consistency(filename, backend):
    """Verify our volume calculations match pydagmc results"""

    # Get volumes from our implementations
    volumes = di.get_volumes_by_cell_id(filename, backend=backend)

    # Get volumes from pydagmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag_model = pydagmc.Model(filename)

    pydagmc_volumes = {
        int(vol_id): float(vol.volume)
        for vol_id, vol in dag_model.volumes_by_id.items()
    }

    assert volumes == pytest.approx(pydagmc_volumes, rel=1e-9, abs=1e-7)


@pytest.mark.parametrize("filename", H5M_TEST_FILES)
@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_triangle_conn_and_coords_pydagmc_consistency(
    filename, backend, canonical_triangles
):
    """Verify our triangle data matches pydagmc results"""

    # Get data from our implementation
    our_data = di.get_triangle_conn_and_coords_by_volume(filename, backend=backend)

    # Get data from pydagmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag_model = pydagmc.Model(filename)

    assert set(our_data) == {int(volume_id) for volume_id in dag_model.volumes_by_id}
    for vol_id, volume in dag_model.volumes_by_id.items():
        vol_id = int(vol_id)

        # Get pydagmc triangle data (with compress=True for unique vertices)
        pydagmc_conn, pydagmc_coords = volume.get_triangle_conn_and_coords(
            compress=True
        )

        # Get our data
        assert vol_id in our_data, f"Volume {vol_id} missing from our implementation"
        our_conn, our_coords = our_data[vol_id]

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
        our_sorted = our_coords[np.lexsort(our_coords[:, ::-1].T)]
        pydagmc_sorted = pydagmc_coords[np.lexsort(pydagmc_coords[:, ::-1].T)]
        np.testing.assert_allclose(
            our_sorted,
            pydagmc_sorted,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Volume {vol_id}: coordinates differ from pydagmc",
        )

        np.testing.assert_allclose(
            canonical_triangles(our_conn, our_coords),
            canonical_triangles(pydagmc_conn, pydagmc_coords),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Volume {vol_id}: triangle geometry differs from pydagmc",
        )
