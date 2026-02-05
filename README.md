
[![N|Python](https://www.python.org/static/community_logos/python-powered-w-100x40.png)](https://www.python.org)

[![CI with install](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/ci_with_install.yml/badge.svg)](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/ci_with_install.yml)

[![codecov](https://codecov.io/gh/fusion-energy/dagmc_h5m_file_inspector/branch/main/graph/badge.svg)](https://codecov.io/gh/fusion-energy/dagmc_h5m_file_inspector)

[![Upload Python Package](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/python-publish.yml/badge.svg?branch=main)](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/python-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/dagmc_h5m_file_inspector?color=brightgreen&label=pypi&logo=grebrightgreenen&logoColor=green)](https://pypi.org/project/dagmc_h5m_file_inspector/)

# dagmc-h5m-file-inspector

A minimal Python package that inspects DAGMC h5m files to extract volume IDs,
material tags, bounding boxes, and geometric volumes.


# Installation

```bash
pip install dagmc-h5m-file-inspector
```

The package uses h5py as the default backend. Optionally, pymoab can be used
as an alternative backend if installed.


# Python API Usage

## Finding volume IDs

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_from_h5m("dagmc.h5m")

>>> [1, 2]
```

## Finding material tags

```python
import dagmc_h5m_file_inspector as di

di.get_materials_from_h5m("dagmc.h5m")

>>> ['big_box', 'small_box']
```

## Finding volume IDs with their materials

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_and_materials_from_h5m("dagmc.h5m")

>>> {1: 'small_box', 2: 'big_box'}
```

## Getting the bounding box

```python
import dagmc_h5m_file_inspector as di

lower_left, upper_right = di.get_bounding_box_from_h5m("dagmc.h5m")

>>> lower_left
array([-5., -10., -10.])

>>> upper_right
array([25., 10., 10.])
```

## Getting geometric volume sizes by cell ID

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_from_h5m_by_cell_id("dagmc.h5m")

>>> {1: 1000.0, 2: 8000.0}
```

## Getting geometric volume sizes by material name

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_from_h5m_by_material_name("dagmc.h5m")

>>> {'small_box': 1000.0, 'big_box': 8000.0}
```

## Getting geometric volume sizes by cell ID and material name

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_from_h5m_by_cell_id_and_material_name("dagmc.h5m")

>>> {(1, 'small_box'): 1000.0, (2, 'big_box'): 8000.0}
```

## Setting OpenMC material volumes from DAGMC geometry

This function reads the DAGMC file, matches materials by name, and sets the
`volume` attribute on the corresponding OpenMC Material objects.

```python
import openmc
import dagmc_h5m_file_inspector as di

# Create OpenMC materials with names matching the DAGMC file
small_box = openmc.Material(name='small_box')
big_box = openmc.Material(name='big_box')
materials = openmc.Materials([small_box, big_box])

# Set volumes from DAGMC geometry
di.set_openmc_material_volumes_from_h5m(materials, "dagmc.h5m")

>>> small_box.volume
1000.0

>>> big_box.volume
8000.0
```

## Getting triangle connectivity and coordinates for each volume

This function extracts the triangle mesh data for each volume, returning the
connectivity (vertex indices) and coordinates (3D points) needed for visualization
or mesh processing.

```python
import dagmc_h5m_file_inspector as di

data = di.get_triangle_conn_and_coords_by_volume("dagmc.h5m")

>>> data
{1: (array([[0, 1, 2], [0, 2, 3], ...]), array([[0., 0., 0.], [10., 0., 0.], ...])),
 2: (array([[0, 1, 2], [0, 2, 3], ...]), array([[-5., -10., -10.], [25., -10., -10.], ...]))}

# Access data for a specific volume
connectivity, coordinates = data[1]
>>> connectivity.shape
(12, 3)  # 12 triangles, each with 3 vertex indices
>>> coordinates.shape
(8, 3)   # 8 unique vertices, each with x, y, z coordinates
```

## Using the pymoab backend

All functions support an optional `backend` parameter. The default is `"h5py"`,
but `"pymoab"` can be used if pymoab is installed:

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_from_h5m("dagmc.h5m", backend="pymoab")

>>> [1, 2]
```
