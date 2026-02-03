from pathlib import Path
from typing import List, Optional

import h5py


def get_volumes_from_h5m(filename: str) -> List[int]:
    """Reads in a DAGMC h5m file and finds the volume ids.

    Arguments:
        filename: the filename of the DAGMC h5m file

    Returns:
        A list of volume ids
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    with h5py.File(filename, "r") as f:
        global_ids = f["tstt/sets/tags/GLOBAL_ID"][()]
        cat_ids = f["tstt/tags/CATEGORY/id_list"][()]
        cat_vals = f["tstt/tags/CATEGORY/values"][()]

        # Build category lookup
        cat_lookup = {}
        for eid, val in zip(cat_ids, cat_vals):
            cat_lookup[int(eid)] = val.tobytes().decode("ascii").rstrip("\x00")

        # Find the base entity ID for sets
        # Sets start after nodes and elements
        base_entity_id = int(cat_ids.min()) - 1

        # Collect volume GLOBAL_IDs
        volume_ids = []
        for i in range(len(global_ids)):
            entity_id = base_entity_id + i
            if cat_lookup.get(entity_id) == "Volume":
                volume_ids.append(int(global_ids[i]))

        return sorted(set(volume_ids))


def get_materials_from_h5m(
    filename: str, remove_prefix: Optional[bool] = True
) -> List[str]:
    """Reads in a DAGMC h5m file and finds the material tags.

    Arguments:
        filename: the filename of the DAGMC h5m file
        remove_prefix: remove the mat: prefix from the material tag or not

    Returns:
        A list of material tags
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

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


def get_volumes_and_materials_from_h5m(
    filename: str, remove_prefix: Optional[bool] = True
) -> dict:
    """Reads in a DAGMC h5m file and finds the volume ids with their
    associated material tags.

    Arguments:
        filename: the filename of the DAGMC h5m file
        remove_prefix: remove the mat: prefix from the material tag or not

    Returns:
        A dictionary of volume ids and material tags
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    with h5py.File(filename, "r") as f:
        global_ids = f["tstt/sets/tags/GLOBAL_ID"][()]
        cat_ids = f["tstt/tags/CATEGORY/id_list"][()]
        cat_vals = f["tstt/tags/CATEGORY/values"][()]
        name_ids = f["tstt/tags/NAME/id_list"][()]
        name_vals = f["tstt/tags/NAME/values"][()]

        # Build category lookup: entity_id -> category string
        cat_lookup = {}
        for eid, val in zip(cat_ids, cat_vals):
            cat_lookup[int(eid)] = val.tobytes().decode("ascii").rstrip("\x00")

        # Build name lookup: entity_id -> name string
        name_lookup = {}
        for eid, val in zip(name_ids, name_vals):
            name_lookup[int(eid)] = val.tobytes().decode("ascii").rstrip("\x00")

        # Find the base entity ID for sets
        base_entity_id = int(cat_ids.min()) - 1

        # Collect volumes: list of (set_idx, GLOBAL_ID)
        volumes = []
        for i in range(len(global_ids)):
            entity_id = base_entity_id + i
            if cat_lookup.get(entity_id) == "Volume":
                volumes.append({"set_idx": i, "gid": int(global_ids[i])})

        # Collect material groups: list of (set_idx, name)
        groups = []
        for i in range(len(global_ids)):
            entity_id = base_entity_id + i
            name = name_lookup.get(entity_id, "")
            if name.startswith("mat:"):
                groups.append({"set_idx": i, "name": name})

        # Sort volumes by GLOBAL_ID (ascending)
        volumes_sorted = sorted(volumes, key=lambda x: x["gid"])
        # Sort groups by set_idx (file order)
        groups_sorted = sorted(groups, key=lambda x: x["set_idx"])

        # Map volumes to materials by pairing in order
        vol_mat = {}
        for vol, grp in zip(volumes_sorted, groups_sorted):
            material_name = grp["name"]
            if remove_prefix:
                material_name = material_name[4:]  # Remove "mat:" prefix
            vol_mat[vol["gid"]] = material_name

        return vol_mat
