
from pathlib import Path
from typing import List, Optional

import pymoab as mb
from pymoab import core, types


def load_moab_file(filename: str):
    """Loads a DAGMC h5m into a Moab Core object and returns the object

    Arguments:
        filename: the filename of the DAGMC h5m file

    Returns:
        A pymoab.core.Core()
    """

    if not Path(filename).is_file():
        msg = f"filename provided ({filename}) does not exist"
        raise FileNotFoundError(msg)

    moab_core = core.Core()
    moab_core.load_file(filename)
    return moab_core


def get_volumes_from_h5m(filename: str) -> List[str]:
    """Reads in a DAGMC h5m file and uses PyMoab to find the volume ids of the
    materials in the file.

    Arguments:
        filename: the filename of the DAGMC h5m file

    Returns:
        A list of volume ids
    """

    # create a new PyMOAB instance and load the specified DAGMC file
    mbcore = load_moab_file(filename)
    group_ents = get_groups(mbcore)
    name_tag = mbcore.tag_get_handle(mb.types.NAME_TAG_NAME)
    id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)
    ids = []

    for group_ent in group_ents:
        group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
        if group_name.startswith('mat:'):
            vols = mbcore.get_entities_by_type(group_ent, mb.types.MBENTITYSET)

            for vol in vols:
                id = mbcore.tag_get_data(id_tag, vol)[0][0]
                ids.append(id)

    return sorted(set(list(ids)))


def get_groups(mbcore):

    category_tag = mbcore.tag_get_handle(mb.types.CATEGORY_TAG_NAME)

    group_category = ["Group"]

    group_ents = mbcore.get_entities_by_type_and_tag(
        0, mb.types.MBENTITYSET, category_tag, group_category)

    return group_ents


def get_materials_from_h5m(
    filename: str,
    remove_prefix: Optional[bool] = True
) -> List[int]:
    """Reads in a DAGMC h5m file and uses PyMoab to find the material tags in
    the file.

    Arguments:
        filename: the filename of the DAGMC h5m file
        remove_prefix: remove the mat: prefix from the material tag or not

    Returns:
        A list of material tags
    """

    mbcore = load_moab_file(filename)
    group_ents = get_groups(mbcore)
    name_tag = mbcore.tag_get_handle(mb.types.NAME_TAG_NAME)

    materials_list = []
    for group_ent in group_ents:

        group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
        if group_name.startswith('mat:'):
            if remove_prefix:
                materials_list.append(group_name[4:])
            else:
                materials_list.append(group_name)

    return sorted(set(materials_list))


def get_vol_mat_map(group_ents, mbcore, remove_prefix) -> dict:
    name_tag = mbcore.tag_get_handle(mb.types.NAME_TAG_NAME)
    id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)
    vol_mat = {}

    for group_ent in group_ents:

        group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
        # optionally confirm that this is a material!

        if group_name.startswith('mat:'):

            vols = mbcore.get_entities_by_type(group_ent, mb.types.MBENTITYSET)

            for vol in vols:
                id = mbcore.tag_get_data(id_tag, vol)[0][0]
                if remove_prefix:
                    vol_mat[id] = group_name[4:]
                else:
                    vol_mat[id] = group_name

    return vol_mat


def get_volumes_and_materials_from_h5m(
    filename: str,
    remove_prefix: Optional[bool] = True
) -> dict:
    """Reads in a DAGMC h5m file and uses PyMoab to find the volume ids with
    their associated material tags.

    Arguments:
        filename: the filename of the DAGMC h5m file
        remove_prefix: remove the mat: prefix from the material tag or not

    Returns:
        A dictionary of volume ids and material tags
    """

    mbcore = load_moab_file(filename)
    group_ents = get_groups(mbcore)
    vol_mat = get_vol_mat_map(group_ents, mbcore, remove_prefix)
    return vol_mat
