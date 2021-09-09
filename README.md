
# dagmc-h5m-file-inspector

A minimal Python package that finds the volume ids and the material tags in a
DAGMC h5m file.


# Installation

```bash
conda install -c conda-forge moab

pip install dagmc-h5m-file-inspector
```


# Usage

Finding the volume IDs in a DAGMC h5m file.

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_from_h5m("dagmc.h5m")

>>> [1, 2]
```

Finding the material tags in a DAGMC h5m file.

```python
import dagmc_h5m_file_inspector as di

di.get_materials_from_h5m("dagmc.h5m")

>>> ['mat:steel', 'mat:graveyard']
```

Finding the volume IDs with their materials present in a DAGMC h5m file.

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_and_materials_from_h5m

>>> {1: 'mat:steel', 2: 'mat:graveyard'}
```

