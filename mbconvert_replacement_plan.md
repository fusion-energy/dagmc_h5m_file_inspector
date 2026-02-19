# Plan: `convert_h5m_to_vtkhdf()` — mbconvert replacement using h5py

## Goal

Add a function that converts a DAGMC `.h5m` file into a `.vtkhdf` file (VTKHDF
UnstructuredGrid format) that can be opened directly in ParaView (5.13+). This
replaces the need for `mbconvert` + VTK dependency. Only `h5py` and `numpy` are
needed — both are already dependencies.

## Background & References

- The VTKHDF format stores VTK data structures inside HDF5 files. The
  `UnstructuredGrid` variant is documented at
  https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html
- OpenMC merged a similar approach in PR #3252 — writing VTKHDF with h5py only,
  no VTK dependency. The user shared that code as reference.
- ParaView 5.13.0+ can read `.vtkhdf` files natively.

## VTKHDF UnstructuredGrid schema (what we need to write)

```
/VTKHDF                                   (Group)
  @Version = [2, 1]                       (Attribute: int64[2])
  @Type    = "UnstructuredGrid"           (Attribute: ASCII string)

  NumberOfPoints          int64[1]         — total vertex count
  NumberOfCells           int64[1]         — total triangle count
  NumberOfConnectivityIds int64[1]         — total_triangles * 3

  Points                  float64[N, 3]   — all vertex coordinates
  Connectivity            int64[M*3]      — flattened triangle vertex indices
  Offsets                 int64[M+1]      — [0, 3, 6, 9, …, M*3]
  Types                   uint8[M]        — all set to 5 (VTK_TRIANGLE)

  CellData/                               (Group)
    cell_id               int32[M]        — DAGMC volume ID per triangle
    material_id           int32[M]        — integer material index per triangle
    material_name         (see below)     — material name string per triangle (optional, see discussion)

  FieldData/                              (Group)
    material_names        variable-length string array — maps material_id → name
```

## Data flow

The package already has everything we need to read from the h5m:

1. **`get_triangle_conn_and_coords_by_volume(filename)`** — returns
   `{vol_id: (connectivity[n_tri,3], coordinates[n_vert,3])}` per volume. Each
   volume's connectivity is 0-based relative to its own coordinate array.

2. **`get_volumes_and_materials_from_h5m(filename)`** — returns
   `{vol_id: material_name}`.

To build the combined VTKHDF mesh we merge per-volume data:

```
global_points = []      # will become (total_verts, 3)
global_conn   = []      # will become (total_tris * 3,)
cell_ids      = []      # one int per triangle
material_ids  = []      # one int per triangle

point_offset = 0
for vol_id in sorted(per_volume_data):
    conn, coords = per_volume_data[vol_id]
    global_points.append(coords)
    global_conn.append(conn + point_offset)     # shift indices
    cell_ids.extend([vol_id] * len(conn))
    material_ids.extend([mat_name_to_int[vol_mat[vol_id]]] * len(conn))
    point_offset += len(coords)
```

Note: vertices shared between volumes will be duplicated. This is fine —
`mbconvert` does the same, and it keeps the logic simple and correct. Each
volume's surface mesh is self-contained.

## API design

```python
def convert_h5m_to_vtkhdf(
    h5m_filename: str,
    vtkhdf_filename: str = "",
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> str:
    """Convert a DAGMC h5m file to a VTKHDF file for visualization in ParaView.

    The output file contains the triangle surface mesh for all volumes, with
    per-triangle cell_id (DAGMC volume ID) and material_id as cell data. This
    allows coloring by volume or material in ParaView.

    Arguments:
        h5m_filename: path to the input DAGMC h5m file
        vtkhdf_filename: path for the output vtkhdf file. If empty string,
            uses the same name as h5m_filename with a .vtkhdf extension.
        backend: backend for reading the h5m file ("h5py" or "pymoab")

    Returns:
        The path to the written vtkhdf file.
    """
```

If `vtkhdf_filename` is an empty string, default to `Path(h5m_filename).with_suffix(".vtkhdf")`.

## Implementation steps

### Step 1: Add `_write_vtkhdf()` internal function to `core.py`

A pure function that takes the merged arrays and writes the HDF5 file:

```python
def _write_vtkhdf(
    filename: str,
    points: np.ndarray,         # (N, 3) float64
    connectivity: np.ndarray,   # (M, 3) int64, 0-based
    cell_ids: np.ndarray,       # (M,) int32
    material_ids: np.ndarray,   # (M,) int32
    material_names: list[str],  # maps material_id index → name
) -> None:
```

This writes:
- `/VTKHDF` group with `Version` and `Type` attributes
- `NumberOfPoints`, `NumberOfCells`, `NumberOfConnectivityIds` (single-partition, each shape `(1,)`)
- `Points` — the coordinates
- `Connectivity` — flattened connectivity
- `Offsets` — `np.arange(0, n_cells * 3 + 1, 3)`
- `Types` — all `5` (`VTK_TRIANGLE`)
- `CellData/cell_id` — DAGMC volume ID per triangle
- `CellData/material_id` — integer material index per triangle
- `FieldData/material_names` — variable-length string array so the user can look up what each `material_id` means

### Step 2: Add `convert_h5m_to_vtkhdf()` public function to `core.py`

This orchestrates the conversion:

1. Validate input file exists
2. Call `get_triangle_conn_and_coords_by_volume(h5m_filename, backend)`
3. Call `get_volumes_and_materials_from_h5m(h5m_filename, backend)`
4. Build unique material name → integer index mapping
5. Merge per-volume meshes into global arrays (offsetting vertex indices)
6. Call `_write_vtkhdf()` with the merged data
7. Return the output filename

### Step 3: Export from `__init__.py`

Add `from .core import convert_h5m_to_vtkhdf` to `__init__.py`.

### Step 4: Add tests

Add tests in `tests/test_python_api_usage.py`:

1. **Round-trip test**: Convert an h5m file, read back the vtkhdf with h5py,
   verify the HDF5 structure has the right groups/datasets/attributes.
2. **Data integrity test**: Verify total triangle and vertex counts match what
   `get_triangle_conn_and_coords_by_volume()` returns.
3. **Cell data test**: Verify `cell_id` and `material_id` arrays have correct
   length and expected values.
4. **Default filename test**: Verify that omitting `vtkhdf_filename` produces a
   file with `.vtkhdf` extension.
5. **Multi-volume test**: Use a test geometry with multiple volumes/materials and
   verify the mapping is correct.

## Files to modify

| File | Change |
|------|--------|
| `src/dagmc_h5m_file_inspector/core.py` | Add `_write_vtkhdf()` and `convert_h5m_to_vtkhdf()` |
| `src/dagmc_h5m_file_inspector/__init__.py` | Export `convert_h5m_to_vtkhdf` |
| `tests/test_python_api_usage.py` | Add conversion tests |

## No new dependencies

`h5py` and `numpy` are already required. Nothing else is needed.

## Verification plan

After implementation:
1. Run the existing test suite to make sure nothing is broken
2. Run the new tests
3. Manually verify by converting `dagmc.h5m` and inspecting the output file's
   HDF5 structure with `h5dump` or `h5py`
