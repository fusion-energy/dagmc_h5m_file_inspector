
[![N|Python](https://www.python.org/static/community_logos/python-powered-w-100x40.png)](https://www.python.org)

[![CI with install](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/ci_with_install.yml/badge.svg)](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/ci_with_install.yml)

[![codecov](https://codecov.io/gh/fusion-energy/dagmc_h5m_file_inspector/branch/main/graph/badge.svg)](https://codecov.io/gh/fusion-energy/dagmc_h5m_file_inspector)

[![Upload Python Package](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/python-publish.yml/badge.svg?branch=main)](https://github.com/fusion-energy/dagmc_h5m_file_inspector/actions/workflows/python-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/dagmc_h5m_file_inspector?color=brightgreen&label=pypi&logo=grebrightgreenen&logoColor=green)](https://pypi.org/project/dagmc_h5m_file_inspector/)

# dagmc-h5m-file-inspector

A minimal Python package that finds the volume ids and the material tags in a
DAGMC h5m file.


# Installation

The dagmc-h5m-file-inspector package requires pymoab which can be installed
alongside Moab with a conda install command. Moab is not avialable on pip,
however it can be installed with Conda.

```bash
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

di.get_volumes_and_materials_from_h5m("dagmc.h5m")

>>> {1: 'steel', 2: 'graveyard'}
```
