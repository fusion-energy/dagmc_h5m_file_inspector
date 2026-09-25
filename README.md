
[![N|Python](https://www.python.org/static/community_logos/python-powered-w-100x40.png)](https://www.python.org)

[![CI with install](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/ci_with_install.yml/badge.svg)](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/ci_with_install.yml)

[![codecov](https://codecov.io/gh/fusion-energy/dagmc_h5m_file_inspector/branch/main/graph/badge.svg)](https://codecov.io/gh/fusion-energy/dagmc_h5m_file_inspector)

[![Upload Python Package](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/python-publish.yml/badge.svg)](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/python-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/dagmc_h5m_file_inspector?color=brightgreen&label=pypi&logo=grebrightgreenen&logoColor=green)](https://pypi.org/project/dagmc_h5m_file_inspector/)

# dagmc-h5m-file-inspector

A minimal Python package that inspects DAGMC h5m files to extract volume IDs,
surface IDs, material tags, bounding boxes, geometric volumes, and surface areas.


# Installation

```bash
pip install dagmc-h5m-file-inspector
```

The package uses h5py as the default backend. Optionally, pymoab can be used
as an alternative backend if installed.


# Python API Usage

## Loading once for multiple queries

Use ``DAGMCFile`` for queries and edits. Geometry and material data are loaded
once and reused by the in-memory methods.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")

volume_ids = dagmc.get_volumes()
materials = dagmc.get_materials()
volumes_and_materials = dagmc.get_volumes_and_materials()
bounding_box = dagmc.get_bounding_box()
```

Edits are applied in memory, so several operations can be performed after one
load and then written once:

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.remove_volumes(1)
dagmc.remove_materials("big_box")
dagmc.move(x=10.0)
dagmc.rotate_around_axis(axis="z", degrees=45)
dagmc.write("modified.h5m")
```

## Finding volume IDs

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_volumes()

# [1, 2]
```

## Finding material tags

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_materials()

# ['big_box', 'small_box']
```

## Finding volume IDs with their materials

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_volumes_and_materials()

# {1: 'small_box', 2: 'big_box'}
```

## Finding cell IDs by group name

Besides the ``mat:`` material groups, a DAGMC h5m file can contain other groups
(for example ``component:`` groups tagging individual components). These readers
surface that non-material group membership so you can map groups onto cell
(volume) IDs, which is handy for building an ``openmc.CellFilter``. Material
(``mat:``) groups are excluded as they are already available via
``get_volumes_and_materials``.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_cell_ids_by_group_name()

# {'component:small_box': [1], 'component:big_box': [2], 'assembly:all': [1, 2]}
```

The inverse mapping (cell ID to the groups it belongs to) is also available:

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_groups_by_cell_id()

# {1: ['assembly:all', 'component:small_box'],
#  2: ['assembly:all', 'component:big_box']}
```

These can be combined with OpenMC to tally on a component rather than duplicating
material definitions:

```python
import openmc
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
cell_ids = dagmc.get_cell_ids_by_group_name()["component:small_box"]
cell_filter = openmc.CellFilter(cell_ids)
```

## Finding surface IDs

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_surface_ids()

# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

## Finding surface IDs by cell ID

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_surface_ids_by_cell_id(cell_id=1)

# [1, 2, 3, 4, 5, 6]
```

## Finding surface IDs by material name

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_surface_ids_by_material_name(material="small_box")

# [1, 2, 3, 4, 5, 6]
```

## Getting the bounding box

Returns a `BoundingBox` object that is API compatible with OpenMC's `openmc.BoundingBox`.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
bbox = dagmc.get_bounding_box()

# bbox == BoundingBox((-5.0, -10.0, -10.0), (25.0, 10.0, 10.0))
bbox.lower_left  # (-5.0, -10.0, -10.0)
bbox.upper_right  # (25.0, 10.0, 10.0)
bbox.center  # (10.0, 0.0, 0.0)
bbox.volume  # 12000.0
bbox.width  # (30.0, 20.0, 20.0)
bbox.extent
# {'xy': (-5.0, 25.0, -10.0, 10.0),
#  'xz': (-5.0, 25.0, -10.0, 10.0),
#  'yz': (-10.0, 10.0, -10.0, 10.0)}
```

The `BoundingBox` supports indexing, unpacking, containment checks, and set operations:

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
bbox = dagmc.get_bounding_box()

# Unpacking
lower_left, upper_right = bbox

# Indexing
bbox[0]  # (-5.0, -10.0, -10.0)

# Point containment
(0.0, 0.0, 0.0) in bbox  # True

# Intersection and union of two bounding boxes
small_box_bbox = dagmc.get_bounding_box(materials="small_box")
bbox_intersection = bbox & small_box_bbox
bbox_union = bbox | small_box_bbox
```

Optionally filter by material tag to get the bounding box for specific materials:

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")

# Bounding box for a single material
bbox = dagmc.get_bounding_box(materials="small_box")

bbox.lower_left  # (-5.0, -5.0, -5.0)
bbox.upper_right  # (5.0, 5.0, 5.0)

# Bounding box for multiple materials (combined)
bbox = dagmc.get_bounding_box(materials=["small_box", "big_box"])

bbox.lower_left  # (-5.0, -10.0, -10.0)
bbox.upper_right  # (25.0, 10.0, 10.0)
```

## Getting geometric volume sizes by cell ID

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_volumes_by_cell_id()

# {1: 1000.0, 2: 8000.0}
```

## Getting geometric volume sizes by material name

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_volumes_by_material_name()

# {'small_box': 1000.0, 'big_box': 8000.0}
```

## Getting geometric volume sizes by cell ID and material name

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_volumes_by_cell_id_and_material_name()

# {(1, 'small_box'): 1000.0, (2, 'big_box'): 8000.0}
```

## Getting surface areas by cell ID

Returns a list of surface areas, one per DAGMC surface bounding the volume.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_surface_area_by_cell_id(cell_id=1)

# [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
```

## Getting surface areas by material name

Returns a list of surface areas for all DAGMC surfaces bounding volumes
with the given material.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_surface_area_by_material_name(material="small_box")

# [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
```

## Getting surface areas by surface ID

Returns a dictionary mapping each surface ID to its area. Useful for
computing wall loading when combined with surface current tallies.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_surface_area_by_surface_id()

# {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0,
#  7: 100.0, 8: 400.0, 9: 400.0, 10: 400.0, 11: 400.0, 12: 400.0}
```

## Getting surface shared status

Returns a dictionary mapping each surface ID to the cell IDs and materials
that share it. Useful for identifying interfaces between volumes.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_surface_shared_status()

# {1: {'materials': ['small_box'], 'cell_ids': [1]},
#  2: {'materials': ['small_box'], 'cell_ids': [1]},
#  ...
#  7: {'materials': ['small_box', 'big_box'], 'cell_ids': [1, 2]},
#  ...}
```

## Setting OpenMC material volumes from DAGMC geometry

This method reads the DAGMC file, matches materials by name, and sets the
`volume` attribute on the corresponding OpenMC Material objects.

```python
import openmc
import dagmc_h5m_file_inspector as di

# Create OpenMC materials with names matching the DAGMC file
small_box = openmc.Material(name="small_box")
big_box = openmc.Material(name="big_box")
materials = openmc.Materials([small_box, big_box])

# Set volumes from DAGMC geometry
dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.set_openmc_material_volumes(materials)

small_box.volume  # 1000.0
big_box.volume  # 8000.0
```

## Getting triangle connectivity and coordinates for each volume

This method extracts the triangle mesh data for each volume, returning the
connectivity (vertex indices) and coordinates (3D points) needed for visualization
or mesh processing.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
data = dagmc.get_triangle_conn_and_coords_by_volume()

# {1: (array([[0, 1, 2], ...]), array([[0., 0., 0.], ...])),
#  2: (array([[0, 1, 2], ...]), array([[-5., -10., -10.], ...]))}

# Access data for a specific volume
connectivity, coordinates = data[1]
connectivity.shape  # (12, 3): 12 triangles, each with 3 vertex indices
coordinates.shape  # (8, 3): 8 unique vertices, each with x, y, z coordinates
```

## Convert h5m file to vtkhdf

Convert DAGMC h5m files to vtkhdf which can be directly opened in Paraview 5.13+.

The resulting Paraview files have color for cell IDs and material tags present within the h5m file.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.convert_to_vtkhdf("dagmc.vtkhdf")
```

![vtk file from dagmc.h5m](dagmc-converted-to-vtkhdf.png)



## Removing materials from h5m files

Remove one or more materials (and their associated volumes) from a DAGMC h5m file,
writing the result to a new file.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.get_materials()

# ['big_box', 'small_box']

# Remove a single material
dagmc.remove_materials("small_box")
dagmc.write("dagmc_reduced.h5m")

reduced = di.DAGMCFile("dagmc_reduced.h5m")
reduced.get_materials()

# ['big_box']
```

```python
import dagmc_h5m_file_inspector as di

reactor = di.DAGMCFile("reactor.h5m")
reactor.get_materials()

# ['blanket', 'first_wall', 'shield']

# Remove multiple materials
reactor.remove_materials(["blanket", "shield"])
reactor.write("reactor_reduced.h5m")

reduced = di.DAGMCFile("reactor_reduced.h5m")
reduced.get_materials()

# ['first_wall']
```

## Removing volumes from h5m files

Remove one or more volumes by ID while retaining all other volumes. Material
tags remain when they are also used by a surviving volume.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("volume_removal.h5m")
dagmc.get_volumes_and_materials()

# {1: 'steel', 2: 'steel', 3: 'water'}

dagmc.remove_volumes(1)
dagmc.write("dagmc_reduced.h5m")

reduced = di.DAGMCFile("dagmc_reduced.h5m")
reduced.get_volumes_and_materials()

# {2: 'steel', 3: 'water'}
```

## Rotating a DAGMC geometry around an axis

Rotate the mesh coordinates around a coordinate axis and write a new h5m file.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.rotate_around_axis(axis="z", degrees=90)
dagmc.write("dagmc_rotated.h5m")
```

## Moving a DAGMC geometry

Translate (move) the mesh coordinates by an offset and write a new h5m file.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")
dagmc.move(x=10.0, y=0.0, z=0.0)
dagmc.write("dagmc_moved.h5m")
```

## Setting boundary conditions on surfaces

Set a boundary condition (e.g. vacuum, reflective) on a DAGMC surface.
This creates a Group entity in the h5m file that OpenMC reads to apply
the boundary condition during transport.

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m")

# Find surface IDs and their areas
areas = dagmc.get_surface_area_by_surface_id()

# {1: 50.0, 2: 80.0, ...}

# Set the larger surface to vacuum (e.g. outer surface of a shell)
dagmc.set_boundary_condition(
    surface_id=2,
    boundary_condition="vacuum",
    output_filename="dagmc_with_bc.h5m",
)
```

Supported boundary conditions: `"vacuum"`, `"reflective"`.
If `output_filename` is omitted the input file is modified in place.

## Combining multiple DAGMC h5m files

Merge multiple DAGMC h5m files into a single file. Volumes are renumbered
sequentially in the output. It is the caller's responsibility to ensure the
geometries do not overlap.

```python
import dagmc_h5m_file_inspector as di

combined = di.DAGMCFile.combine_h5m_files(
    input_files=["file_a.h5m", "file_b.h5m"],
    output_file="combined.h5m",
)

combined.get_volumes()

# [1, 2]

combined.get_materials()

# ['mat_a', 'mat_b']
```

## Using the pymoab backend

The backend is selected when constructing a `DAGMCFile`. The default is
`"h5py"`, but `"pymoab"` can be used if pymoab is installed:

```python
import dagmc_h5m_file_inspector as di

dagmc = di.DAGMCFile("dagmc.h5m", backend="pymoab")
dagmc.get_volumes()

# [1, 2]
```
