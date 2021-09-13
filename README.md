
# dagmc-h5m-file-inspector

A minimal Python package that finds the volume ids and the material tags in a
DAGMC h5m file.


# Installation

```bash
conda install -c conda-forge moab

pip install dagmc-h5m-file-inspector
```


# Python API Usage

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

>>> ['steel', 'graveyard']
```

Finding the volume IDs with their materials present in a DAGMC h5m file.

```python
import dagmc_h5m_file_inspector as di

di.get_volumes_and_materials_from_h5m

>>> {1: 'steel', 2: 'graveyard'}
```

# Command line tool usage

The options for the command line tool can be obtained with ```inspect-dagmc-h5m-file --help```

Finding the volume IDs present in a dagmc h5m file

```
inspect-dagmc-h5m-file -i dagmc.h5m -v
>>> Volume IDs =[1, 2]
```

Finding the material tags present in a dagmc h5m file

```bash
inspect-dagmc-h5m-file -i dagmc.h5m -m
>>> Material tags =['steel', 'graveyard']
```

Finding the volume IDs and materials present in a dagmc h5m file

```bash
inspect-dagmc-h5m-file -i dagmc.h5m -b
>>> Volume IDs and material tags=
     {   1: 'steel',
         2: 'graveyard'}
```

# Aknowledgements

This package is based on a [Python script](https://gist.github.com/gonuke/c36e327e399c7a685cd315c738121c9a) by @gonuke
