from pathlib import Path
from typing import List, Optional, Literal

import h5py


# ============================================================================
# h5py backend implementation
# ============================================================================


def _get_volumes_h5py(filename: str) -> List[int]:
    """Get volume IDs using h5py backend."""
    with h5py.File(filename, "r") as f:
        global_ids = f["tstt/sets/tags/GLOBAL_ID"][()]
        cat_ids = f["tstt/tags/CATEGORY/id_list"][()]
        cat_vals = f["tstt/tags/CATEGORY/values"][()]

        cat_lookup = {}
        for eid, val in zip(cat_ids, cat_vals):
            cat_lookup[int(eid)] = val.tobytes().decode("ascii").rstrip("\x00")

        base_entity_id = int(cat_ids.min()) - 1

        volume_ids = []
        for i in range(len(global_ids)):
            entity_id = base_entity_id + i
            if cat_lookup.get(entity_id) == "Volume":
                volume_ids.append(int(global_ids[i]))

        return sorted(set(volume_ids))


def _get_materials_h5py(filename: str, remove_prefix: bool) -> List[str]:
    """Get material names using h5py backend."""
    with h5py.File(filename, "r") as f:
        name_ids = f["tstt/tags/NAME/id_list"][()]
        name_vals = f["tstt/tags/NAME/values"][()]

        materials_list = []
        for eid, val in zip(name_ids, name_vals):
            name = val.tobytes().decode("ascii").rstrip("\x00")
            if name.startswith("mat:"):
                if remove_prefix:
                    materials_list.append(name[4:])
                else:
                    materials_list.append(name)

        return sorted(set(materials_list))


def _get_volumes_and_materials_h5py(filename: str, remove_prefix: bool) -> dict:
    """Get volume-to-material mapping using h5py backend."""
    with h5py.File(filename, "r") as f:
        global_ids = f["tstt/sets/tags/GLOBAL_ID"][()]
        cat_ids = f["tstt/tags/CATEGORY/id_list"][()]
        cat_vals = f["tstt/tags/CATEGORY/values"][()]
        name_ids = f["tstt/tags/NAME/id_list"][()]
        name_vals = f["tstt/tags/NAME/values"][()]

        cat_lookup = {}
        for eid, val in zip(cat_ids, cat_vals):
            cat_lookup[int(eid)] = val.tobytes().decode("ascii").rstrip("\x00")

        name_lookup = {}
        for eid, val in zip(name_ids, name_vals):
            name_lookup[int(eid)] = val.tobytes().decode("ascii").rstrip("\x00")

        base_entity_id = int(cat_ids.min()) - 1

        volumes = []
        for i in range(len(global_ids)):
            entity_id = base_entity_id + i
            if cat_lookup.get(entity_id) == "Volume":
                volumes.append({"set_idx": i, "gid": int(global_ids[i])})

        groups = []
        for i in range(len(global_ids)):
            entity_id = base_entity_id + i
            name = name_lookup.get(entity_id, "")
            if name.startswith("mat:"):
                groups.append({"set_idx": i, "name": name})

        volumes_sorted = sorted(volumes, key=lambda x: x["gid"])
        groups_sorted = sorted(groups, key=lambda x: x["set_idx"])

        vol_mat = {}
        for vol, grp in zip(volumes_sorted, groups_sorted):
            material_name = grp["name"]
            if remove_prefix:
                material_name = material_name[4:]
            vol_mat[vol["gid"]] = material_name

        return vol_mat


# ============================================================================
# pymoab backend implementation
# ============================================================================


def _check_pymoab_available():
    """Check if pymoab is available and raise ImportError if not."""
    try:
        import pymoab  # noqa: F401
    except ImportError:
        raise ImportError(
            "pymoab is not installed. Install it to use backend='pymoab', "
            "or use the default h5py backend."
        )


def _load_moab_file(filename: str):
    """Load a DAGMC h5m file into a pymoab Core object."""
    import pymoab as mb
    from pymoab import core

    moab_core = core.Core()
    moab_core.load_file(filename)
    return moab_core


def _get_groups_pymoab(mbcore):
    """Get group entities using pymoab."""
    import pymoab as mb

    category_tag = mbcore.tag_get_handle(mb.types.CATEGORY_TAG_NAME)
    group_category = ["Group"]
    group_ents = mbcore.get_entities_by_type_and_tag(
        0, mb.types.MBENTITYSET, category_tag, group_category
    )
    return group_ents


def _get_volumes_pymoab(filename: str) -> List[int]:
    """Get volume IDs using pymoab backend."""
    import pymoab as mb

    mbcore = _load_moab_file(filename)
    group_ents = _get_groups_pymoab(mbcore)
    name_tag = mbcore.tag_get_handle(mb.types.NAME_TAG_NAME)
    id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)
    ids = []

    for group_ent in group_ents:
        group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
        if group_name.startswith("mat:"):
            vols = mbcore.get_entities_by_type(group_ent, mb.types.MBENTITYSET)
            for vol in vols:
                id = mbcore.tag_get_data(id_tag, vol)[0][0]
                ids.append(id.item())

    return sorted(set(list(ids)))


def _get_materials_pymoab(filename: str, remove_prefix: bool) -> List[str]:
    """Get material names using pymoab backend."""
    import pymoab as mb

    mbcore = _load_moab_file(filename)
    group_ents = _get_groups_pymoab(mbcore)
    name_tag = mbcore.tag_get_handle(mb.types.NAME_TAG_NAME)

    materials_list = []
    for group_ent in group_ents:
        group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
        if group_name.startswith("mat:"):
            if remove_prefix:
                materials_list.append(group_name[4:])
            else:
                materials_list.append(group_name)

    return sorted(set(materials_list))


def _get_volumes_and_materials_pymoab(filename: str, remove_prefix: bool) -> dict:
    """Get volume-to-material mapping using pymoab backend."""
    import pymoab as mb

    mbcore = _load_moab_file(filename)
    group_ents = _get_groups_pymoab(mbcore)
    name_tag = mbcore.tag_get_handle(mb.types.NAME_TAG_NAME)
    id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)
    vol_mat = {}

    for group_ent in group_ents:
        group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
        if group_name.startswith("mat:"):
            vols = mbcore.get_entities_by_type(group_ent, mb.types.MBENTITYSET)
            for vol in vols:
                id = mbcore.tag_get_data(id_tag, vol)[0][0].item()
                if remove_prefix:
                    vol_mat[id] = group_name[4:]
                else:
                    vol_mat[id] = group_name

    return vol_mat


# ============================================================================
# Public API
# ============================================================================


def get_volumes_from_h5m(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> List[int]:
    """Reads in a DAGMC h5m file and finds the volume ids.

    Arguments:
        filename: the filename of the DAGMC h5m file
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A list of volume ids
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_volumes_pymoab(filename)
    else:
        return _get_volumes_h5py(filename)


def get_materials_from_h5m(
    filename: str,
    remove_prefix: Optional[bool] = True,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> List[str]:
    """Reads in a DAGMC h5m file and finds the material tags.

    Arguments:
        filename: the filename of the DAGMC h5m file
        remove_prefix: remove the mat: prefix from the material tag or not
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A list of material tags
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_materials_pymoab(filename, remove_prefix)
    else:
        return _get_materials_h5py(filename, remove_prefix)


def get_volumes_and_materials_from_h5m(
    filename: str,
    remove_prefix: Optional[bool] = True,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> dict:
    """Reads in a DAGMC h5m file and finds the volume ids with their
    associated material tags.

    Arguments:
        filename: the filename of the DAGMC h5m file
        remove_prefix: remove the mat: prefix from the material tag or not
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A dictionary of volume ids and material tags
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_volumes_and_materials_pymoab(filename, remove_prefix)
    else:
        return _get_volumes_and_materials_h5py(filename, remove_prefix)
