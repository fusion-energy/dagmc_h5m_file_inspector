from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal, Sequence, Tuple, Union

import h5py
import numpy as np


RANGE_COMPRESSED_FLAG = 0x8


@dataclass(frozen=True)
class _SetInfo:
    """Internal dataclass for storing MOAB set information."""
    handle: int
    contents: Sequence[int] | Sequence[Tuple[int, int]]
    contents_are_ranges: bool
    children: Sequence[int]
    parents: Sequence[int]
    flags: int


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


def _get_bounding_box_h5py(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounding box using h5py backend."""
    with h5py.File(filename, "r") as f:
        coords = f["tstt/nodes/coordinates"][()]
        lower_left = coords.min(axis=0)
        upper_right = coords.max(axis=0)
        return lower_left, upper_right


def _calculate_triangle_volumes(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Calculate the volume enclosed by a triangular mesh using signed tetrahedra.

    For a closed mesh, the sum of signed tetrahedra volumes (formed by each
    triangle and the origin) gives the enclosed volume.
    """
    # Get vertices for each triangle
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Signed volume of tetrahedron = (v0 · (v1 × v2)) / 6
    cross = np.cross(v1, v2)
    signed_volumes = np.sum(v0 * cross, axis=1) / 6.0

    return abs(np.sum(signed_volumes))


# ============================================================================
# h5py volume calculation helpers
# ============================================================================


def _read_nodes_h5py(f: h5py.File) -> Tuple[np.ndarray, int]:
    """Read node coordinates and start ID from h5py file."""
    nodes = f["tstt/nodes/coordinates"]
    coords = nodes[...]
    node_start = int(nodes.attrs["start_id"])
    return coords, node_start


def _read_tri3_connectivity_h5py(f: h5py.File) -> Tuple[np.ndarray, int]:
    """Read triangle connectivity and start ID from h5py file."""
    tri = f["tstt/elements/Tri3/connectivity"]
    tri_conn = tri[...]
    tri_start = int(tri.attrs["start_id"])
    return tri_conn, tri_start


def _slices_from_end_indices(ends: np.ndarray) -> List[Optional[slice]]:
    """Convert end indices to slices."""
    prev_end = -1
    slices: List[Optional[slice]] = []
    for end in ends.tolist():
        start = prev_end + 1
        if end >= start:
            slices.append(slice(start, end + 1))
        else:
            slices.append(None)
        prev_end = end
    return slices


def _read_sets_h5py(f: h5py.File) -> List[_SetInfo]:
    """Read all entity sets from h5py file."""
    list_ds = f["tstt/sets/list"]
    list_arr = list_ds[...]
    start_id = int(list_ds.attrs["start_id"])
    contents = f["tstt/sets/contents"][...]
    children = f["tstt/sets/children"][...]
    parents = f["tstt/sets/parents"][...]

    contents_slices = _slices_from_end_indices(list_arr[:, 0])
    children_slices = _slices_from_end_indices(list_arr[:, 1])
    parents_slices = _slices_from_end_indices(list_arr[:, 2])

    sets: List[_SetInfo] = []
    for idx in range(list_arr.shape[0]):
        handle = start_id + idx
        flags = int(list_arr[idx, 3])

        contents_slice = contents_slices[idx]
        if contents_slice is None:
            contents_data: Sequence[int] | Sequence[Tuple[int, int]] = []
            contents_are_ranges = False
        else:
            data = contents[contents_slice]
            if flags & RANGE_COMPRESSED_FLAG:
                if len(data) % 2 != 0:
                    raise ValueError(
                        f"Range-compressed contents for set {handle} "
                        f"has odd length {len(data)}"
                    )
                contents_data = [
                    (int(data[i]), int(data[i + 1]))
                    for i in range(0, len(data), 2)
                ]
                contents_are_ranges = True
            else:
                contents_data = [int(v) for v in data]
                contents_are_ranges = False

        children_slice = children_slices[idx]
        if children_slice is None:
            child_list: Sequence[int] = []
        else:
            child_list = [int(v) for v in children[children_slice]]

        parents_slice = parents_slices[idx]
        if parents_slice is None:
            parent_list: Sequence[int] = []
        else:
            parent_list = [int(v) for v in parents[parents_slice]]

        sets.append(
            _SetInfo(
                handle=handle,
                contents=contents_data,
                contents_are_ranges=contents_are_ranges,
                children=child_list,
                parents=parent_list,
                flags=flags,
            )
        )

    return sets


def _read_tag_h5py(f: h5py.File, tag_name: str) -> Dict[int, object]:
    """Read a tag from h5py file and return handle -> value mapping."""
    try:
        tag_group = f[f"tstt/tags/{tag_name}"]
    except KeyError:
        return {}

    if "id_list" not in tag_group or "values" not in tag_group:
        return {}

    ids = tag_group["id_list"][...]
    values = tag_group["values"][...]

    decoded: Dict[int, object] = {}
    if values.dtype.kind in {"S", "V"}:
        for h, v in zip(ids, values):
            if hasattr(v, "tobytes"):
                data = v.tobytes()
            else:
                data = bytes(v)
            decoded[int(h)] = data.split(b"\x00", 1)[0].decode("ascii", "replace")
    else:
        for h, v in zip(ids, values):
            decoded[int(h)] = int(v) if np.issubdtype(values.dtype, np.integer) else v

    return decoded


def _read_geom_sense_h5py(f: h5py.File) -> Dict[int, Tuple[int, int]]:
    """Read GEOM_SENSE_2 tag from h5py file."""
    try:
        tag_group = f["tstt/tags/GEOM_SENSE_2"]
    except KeyError:
        return {}

    if "id_list" not in tag_group or "values" not in tag_group:
        return {}

    ids = tag_group["id_list"][...]
    values = tag_group["values"][...]
    return {
        int(h): (int(v[0]), int(v[1]))
        for h, v in zip(ids, values)
    }


def _expand_set_contents(
    set_info: _SetInfo,
    target_min: Optional[int] = None,
    target_max: Optional[int] = None,
) -> List[int]:
    """Expand set contents, handling range compression."""
    if not set_info.contents:
        return []

    if not set_info.contents_are_ranges:
        return [int(v) for v in set_info.contents]

    handles: List[int] = []
    for start, count in set_info.contents:
        end = start + count - 1
        if target_min is not None:
            start = max(start, target_min)
        if target_max is not None:
            end = min(end, target_max)
        if start <= end:
            handles.extend(range(start, end + 1))
    return handles


def _surface_sign_for_volume(
    vol_handle: int,
    sense: Optional[Tuple[int, int]],
) -> float:
    """Determine surface sign (+1 or -1) relative to a volume."""
    if sense is None:
        return 1.0
    forward, reverse = sense
    if vol_handle == forward and vol_handle != reverse:
        return 1.0
    if vol_handle == reverse and vol_handle != forward:
        return -1.0
    return 1.0


def _tri_indices_for_set(
    set_info: _SetInfo,
    *,
    tri_start: int,
    tri_end: int,
) -> np.ndarray:
    """Get triangle indices (0-based) for a set."""
    if not set_info.contents:
        return np.array([], dtype=np.int64)

    if set_info.contents_are_ranges:
        indices: List[int] = []
        for start, count in set_info.contents:
            end = start + count - 1
            if end < tri_start or start > tri_end:
                continue
            start = max(start, tri_start)
            end = min(end, tri_end)
            indices.extend(range(start - tri_start, end - tri_start + 1))
        return np.asarray(indices, dtype=np.int64)

    handles = [
        h for h in set_info.contents
        if tri_start <= h <= tri_end
    ]
    if not handles:
        return np.array([], dtype=np.int64)
    return np.asarray(handles, dtype=np.int64) - tri_start


def _signed_volume_from_tris(
    coords: np.ndarray,
    tri_conn0: np.ndarray,
    tri_indices: np.ndarray,
) -> float:
    """Calculate signed volume from triangles using tetrahedra method."""
    tri_nodes = tri_conn0[tri_indices]
    v0 = coords[tri_nodes[:, 0]]
    v1 = coords[tri_nodes[:, 1]]
    v2 = coords[tri_nodes[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def _volume_for_volume_set(
    *,
    vol_handle: int,
    sets_by_handle: Dict[int, _SetInfo],
    surface_handles: set,
    geom_sense: Dict[int, Tuple[int, int]],
    coords: np.ndarray,
    tri_conn0: np.ndarray,
    tri_start: int,
    tri_end: int,
) -> float:
    """Calculate the geometric volume for a single volume entity."""
    volume_set = sets_by_handle.get(vol_handle)
    if volume_set is None:
        return 0.0

    if volume_set.children:
        surfaces = [h for h in volume_set.children if h in surface_handles]
    else:
        surfaces = [
            h for h in surface_handles
            if h in geom_sense and vol_handle in geom_sense[h]
        ]

    total = 0.0
    for surf_handle in surfaces:
        surf_set = sets_by_handle.get(surf_handle)
        if surf_set is None:
            continue

        sense = geom_sense.get(surf_handle)
        sign = _surface_sign_for_volume(vol_handle, sense)

        tri_indices = _tri_indices_for_set(
            surf_set,
            tri_start=tri_start,
            tri_end=tri_end,
        )
        if tri_indices.size == 0:
            continue

        total += sign * _signed_volume_from_tris(
            coords,
            tri_conn0,
            tri_indices,
        )

    return total


def _get_volumes_sizes_h5py(filename: str) -> dict:
    """Get geometric volume sizes for each volume ID using h5py backend.

    Uses the parent-child relationships (Volume -> Surfaces) and GEOM_SENSE_2
    to properly assign surfaces to volumes with correct orientation.
    """
    with h5py.File(filename, "r") as f:
        coords, node_start = _read_nodes_h5py(f)
        tri_conn, tri_start = _read_tri3_connectivity_h5py(f)
        tri_conn0 = tri_conn - node_start
        tri_end = tri_start + tri_conn.shape[0] - 1

        sets = _read_sets_h5py(f)
        sets_by_handle = {s.handle: s for s in sets}

        categories = _read_tag_h5py(f, "CATEGORY")
        geom_dim = _read_tag_h5py(f, "GEOM_DIMENSION")
        geom_sense = _read_geom_sense_h5py(f)

        # Get GLOBAL_ID for sets - this can be stored as:
        # 1. Dense array in tstt/sets/tags/GLOBAL_ID
        # 2. Sparse tag in tstt/tags/GLOBAL_ID with id_list/values
        global_ids: Dict[int, int] = {}

        # Try dense array first (more common)
        sets_start_id = int(f["tstt/sets/list"].attrs["start_id"])
        if "tstt/sets/tags/GLOBAL_ID" in f:
            dense_gids = f["tstt/sets/tags/GLOBAL_ID"][...]
            for idx, gid in enumerate(dense_gids):
                handle = sets_start_id + idx
                global_ids[handle] = int(gid)
        else:
            # Fall back to sparse tag
            global_ids = _read_tag_h5py(f, "GLOBAL_ID")

        # Build set of surface handles
        surface_handles = {
            h
            for h, cat in categories.items()
            if cat == "Surface"
        }
        surface_handles.update(
            h for h, dim in geom_dim.items() if dim == 2
        )

        # Build set of volume handles
        volume_handles = {
            h
            for h, cat in categories.items()
            if cat == "Volume"
        }
        volume_handles.update(
            h for h, dim in geom_dim.items() if dim == 3
        )

        volume_sizes = {}
        for vol_handle in volume_handles:
            vol_gid = global_ids.get(vol_handle)
            if vol_gid is None:
                continue

            size = _volume_for_volume_set(
                vol_handle=vol_handle,
                sets_by_handle=sets_by_handle,
                surface_handles=surface_handles,
                geom_sense=geom_sense,
                coords=coords,
                tri_conn0=tri_conn0,
                tri_start=tri_start,
                tri_end=tri_end,
            )
            volume_sizes[int(vol_gid)] = abs(size)

        return volume_sizes


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


def _get_bounding_box_pymoab(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounding box using pymoab backend."""
    import pymoab as mb

    mbcore = _load_moab_file(filename)
    # Get all vertices
    vertices = mbcore.get_entities_by_type(0, mb.types.MBVERTEX)
    coords = mbcore.get_coords(vertices)
    coords = coords.reshape(-1, 3)

    lower_left = coords.min(axis=0)
    upper_right = coords.max(axis=0)
    return lower_left, upper_right


def _get_triangle_conn_and_coords_h5py(filename: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Get triangle connectivity and coordinates for each volume using h5py backend.

    Returns a dictionary mapping volume IDs to tuples of (connectivity, coordinates)
    where connectivity is an Mx3 array of vertex indices and coordinates is an Nx3
    array of 3D points. The connectivity indices are 0-based relative to the
    coordinates array for that volume.
    """
    with h5py.File(filename, "r") as f:
        coords, node_start = _read_nodes_h5py(f)
        tri_conn, tri_start = _read_tri3_connectivity_h5py(f)
        tri_conn0 = tri_conn - node_start  # Convert to 0-based indexing
        tri_end = tri_start + tri_conn.shape[0] - 1

        sets = _read_sets_h5py(f)
        sets_by_handle = {s.handle: s for s in sets}

        categories = _read_tag_h5py(f, "CATEGORY")
        geom_dim = _read_tag_h5py(f, "GEOM_DIMENSION")

        # Get GLOBAL_ID for sets
        global_ids: Dict[int, int] = {}
        sets_start_id = int(f["tstt/sets/list"].attrs["start_id"])
        if "tstt/sets/tags/GLOBAL_ID" in f:
            dense_gids = f["tstt/sets/tags/GLOBAL_ID"][...]
            for idx, gid in enumerate(dense_gids):
                handle = sets_start_id + idx
                global_ids[handle] = int(gid)
        else:
            global_ids = _read_tag_h5py(f, "GLOBAL_ID")

        # Build set of surface handles
        surface_handles = {
            h for h, cat in categories.items() if cat == "Surface"
        }
        surface_handles.update(
            h for h, dim in geom_dim.items() if dim == 2
        )

        # Build set of volume handles
        volume_handles = {
            h for h, cat in categories.items() if cat == "Volume"
        }
        volume_handles.update(
            h for h, dim in geom_dim.items() if dim == 3
        )

        result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for vol_handle in volume_handles:
            vol_gid = global_ids.get(vol_handle)
            if vol_gid is None:
                continue

            volume_set = sets_by_handle.get(vol_handle)
            if volume_set is None:
                continue

            # Get child surfaces of this volume
            if volume_set.children:
                surfaces = [h for h in volume_set.children if h in surface_handles]
            else:
                # Fallback: find surfaces that reference this volume
                surfaces = list(surface_handles)

            # Collect all triangle indices for this volume
            all_tri_indices: List[int] = []
            for surf_handle in surfaces:
                surf_set = sets_by_handle.get(surf_handle)
                if surf_set is None:
                    continue

                tri_indices = _tri_indices_for_set(
                    surf_set,
                    tri_start=tri_start,
                    tri_end=tri_end,
                )
                all_tri_indices.extend(tri_indices.tolist())

            if not all_tri_indices:
                # Empty volume
                result[int(vol_gid)] = (
                    np.array([], dtype=np.int64).reshape(0, 3),
                    np.array([], dtype=np.float64).reshape(0, 3),
                )
                continue

            all_tri_indices = np.array(all_tri_indices, dtype=np.int64)

            # Get the triangles for this volume
            volume_tris = tri_conn0[all_tri_indices]

            # Find unique vertex indices and create local indexing
            unique_verts = np.unique(volume_tris)
            vert_to_local = {v: i for i, v in enumerate(unique_verts)}

            # Extract coordinates for these vertices
            volume_coords = coords[unique_verts]

            # Re-index connectivity to be 0-based relative to volume_coords
            local_conn = np.array(
                [[vert_to_local[v] for v in tri] for tri in volume_tris],
                dtype=np.int64,
            )

            result[int(vol_gid)] = (local_conn, volume_coords)

        return result


def _get_triangle_conn_and_coords_pymoab(filename: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Get triangle connectivity and coordinates for each volume using pymoab backend.

    Returns a dictionary mapping volume IDs to tuples of (connectivity, coordinates)
    where connectivity is an Mx3 array of vertex indices and coordinates is an Nx3
    array of 3D points. The connectivity indices are 0-based relative to the
    coordinates array for that volume.
    """
    import pymoab as mb
    from pymoab import types

    mbcore = _load_moab_file(filename)
    category_tag = mbcore.tag_get_handle(mb.types.CATEGORY_TAG_NAME)
    id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)

    # Get all volumes
    volume_ents = mbcore.get_entities_by_type_and_tag(
        0, mb.types.MBENTITYSET, category_tag, ["Volume"]
    )

    result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for vol_ent in volume_ents:
        vol_gid = mbcore.tag_get_data(id_tag, vol_ent)[0][0].item()

        # Get child surfaces of this volume
        surfaces = mbcore.get_child_meshsets(vol_ent)

        # Collect all triangles and vertices for this volume
        all_verts = set()
        all_tris_conn = []

        for surf in surfaces:
            tris = mbcore.get_entities_by_type(surf, mb.types.MBTRI)
            for tri in tris:
                conn = mbcore.get_connectivity(tri)
                all_verts.update(conn)
                all_tris_conn.append(list(conn))

        if not all_tris_conn:
            result[vol_gid] = (
                np.array([], dtype=np.int64).reshape(0, 3),
                np.array([], dtype=np.float64).reshape(0, 3),
            )
            continue

        # Create local vertex indexing
        all_verts = list(all_verts)
        vert_to_local = {v: i for i, v in enumerate(all_verts)}

        # Get coordinates
        volume_coords = mbcore.get_coords(all_verts).reshape(-1, 3)

        # Re-index connectivity to be 0-based relative to volume_coords
        local_conn = np.array(
            [[vert_to_local[v] for v in tri] for tri in all_tris_conn],
            dtype=np.int64,
        )

        result[vol_gid] = (local_conn, volume_coords)

    return result


def _get_volumes_sizes_pymoab(filename: str) -> dict:
    """Get geometric volume sizes for each volume ID using pymoab backend.

    Uses GEOM_SENSE_2 tag to determine surface orientation relative to each
    volume, enabling correct signed volume calculation for nested geometries.
    """
    import pymoab as mb
    from pymoab import types

    mbcore = _load_moab_file(filename)
    category_tag = mbcore.tag_get_handle(mb.types.CATEGORY_TAG_NAME)
    id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)

    # Get the GEOM_SENSE_2 tag - this stores [forward_vol, reverse_vol] for each surface
    try:
        geom_sense_tag = mbcore.tag_get_handle("GEOM_SENSE_2")
    except RuntimeError:
        geom_sense_tag = None

    # Get all volumes
    volume_ents = mbcore.get_entities_by_type_and_tag(
        0, mb.types.MBENTITYSET, category_tag, ["Volume"]
    )

    volume_sizes = {}

    for vol_ent in volume_ents:
        vol_gid = mbcore.tag_get_data(id_tag, vol_ent)[0][0].item()

        # Get child surfaces of this volume
        surfaces = mbcore.get_child_meshsets(vol_ent)

        total_signed_volume = 0.0

        for surf in surfaces:
            # Determine the sign for this surface relative to this volume
            sign = 1.0
            if geom_sense_tag is not None:
                try:
                    sense_data = mbcore.tag_get_data(geom_sense_tag, surf)
                    # sense_data is [forward_vol, reverse_vol]
                    forward_vol = sense_data[0][0]
                    reverse_vol = sense_data[0][1]
                    if vol_ent == forward_vol and vol_ent != reverse_vol:
                        sign = 1.0
                    elif vol_ent == reverse_vol and vol_ent != forward_vol:
                        sign = -1.0
                    # If vol_ent equals both or neither, default to +1
                except RuntimeError:
                    pass  # Tag not set for this surface, use default sign

            # Get triangles in this surface
            tris = mbcore.get_entities_by_type(surf, mb.types.MBTRI)

            if not tris:
                continue

            # Get all unique vertices for this surface's triangles
            all_verts = set()
            for tri in tris:
                conn = mbcore.get_connectivity(tri)
                all_verts.update(conn)

            all_verts = list(all_verts)
            vert_to_idx = {v: i for i, v in enumerate(all_verts)}

            # Get coordinates
            coords = mbcore.get_coords(all_verts).reshape(-1, 3)

            # Build triangle array with local indices
            tri_array = []
            for tri in tris:
                conn = mbcore.get_connectivity(tri)
                tri_array.append([vert_to_idx[v] for v in conn])
            tri_array = np.array(tri_array)

            # Calculate signed volume for this surface's triangles
            v0 = coords[tri_array[:, 0]]
            v1 = coords[tri_array[:, 1]]
            v2 = coords[tri_array[:, 2]]
            cross = np.cross(v1, v2)
            surface_signed_volume = np.sum(v0 * cross, axis=1).sum() / 6.0

            total_signed_volume += sign * surface_signed_volume

        volume_sizes[vol_gid] = abs(total_signed_volume)

    return volume_sizes


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


def get_bounding_box_from_h5m(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads in a DAGMC h5m file and returns the axis-aligned bounding box.

    Arguments:
        filename: the filename of the DAGMC h5m file
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A tuple of (lower_left, upper_right) numpy arrays representing
        the corners of the bounding box
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_bounding_box_pymoab(filename)
    else:
        return _get_bounding_box_h5py(filename)


def get_volumes_sizes_from_h5m_by_cell_id(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> dict:
    """Reads in a DAGMC h5m file and calculates the geometric volume
    (size) of each volume entity.

    Arguments:
        filename: the filename of the DAGMC h5m file
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A dictionary mapping volume IDs (cell IDs) to their geometric volumes (sizes)
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_volumes_sizes_pymoab(filename)
    else:
        return _get_volumes_sizes_h5py(filename)


def get_volumes_sizes_from_h5m_by_material_name(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> Dict[str, float]:
    """Reads in a DAGMC h5m file and calculates the geometric volume
    for each material, aggregating volumes from all cells with the same material.

    Arguments:
        filename: the filename of the DAGMC h5m file
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A dictionary mapping material names to their total geometric volumes.
        If a material is assigned to multiple cells, their volumes are summed.
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    # Get volume-to-material mapping and volume sizes
    vol_mat_mapping = get_volumes_and_materials_from_h5m(
        filename=filename,
        remove_prefix=True,
        backend=backend,
    )
    volume_sizes = get_volumes_sizes_from_h5m_by_cell_id(
        filename=filename,
        backend=backend,
    )

    # Aggregate volumes by material name
    material_volumes: Dict[str, float] = {}
    for vol_id, mat_name in vol_mat_mapping.items():
        if mat_name not in material_volumes:
            material_volumes[mat_name] = 0.0
        material_volumes[mat_name] += volume_sizes.get(vol_id, 0.0)

    return material_volumes


def set_openmc_material_volumes_from_h5m(
    materials: Union[List, "openmc.Materials"],
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> None:
    """Sets the volume attribute on OpenMC Material objects based on DAGMC geometry.

    This function reads volume and material information from a DAGMC h5m file,
    then matches materials by name and sets the `volume` attribute on the
    corresponding OpenMC Material objects.

    If a material name in the DAGMC file appears in multiple volumes, the
    geometric volumes are summed together.

    Arguments:
        materials: A list of openmc.Material objects or an openmc.Materials
            collection. Materials are matched by their `name` attribute.
        filename: The filename of the DAGMC h5m file.
        backend: The backend to use for reading the file ("h5py" or "pymoab").
            Note: "pymoab" backend is required for accurate volume calculations.

    Raises:
        FileNotFoundError: If the DAGMC file does not exist.
        ValueError: If multiple OpenMC materials have the same name.

    Example:
        >>> import openmc
        >>> steel = openmc.Material(name='steel')
        >>> water = openmc.Material(name='water')
        >>> materials = openmc.Materials([steel, water])
        >>> set_openmc_material_volumes_from_h5m(materials, 'dagmc.h5m')
        >>> print(steel.volume)  # Volume is now set
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    # Check for duplicate material names in the provided materials
    material_names = [mat.name for mat in materials]
    seen_names = {}
    for name in material_names:
        if name is None:
            continue
        if name in seen_names:
            raise ValueError(
                f"Multiple OpenMC materials have the same name '{name}'. "
                "Each material must have a unique name for matching."
            )
        seen_names[name] = True

    # Get volumes aggregated by material name
    material_volumes = get_volumes_sizes_from_h5m_by_material_name(
        filename=filename,
        backend=backend,
    )

    # Set volumes on matching OpenMC materials
    for mat in materials:
        if mat.name is not None and mat.name in material_volumes:
            mat.volume = material_volumes[mat.name]


def get_triangle_conn_and_coords_by_volume(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Reads a DAGMC h5m file and extracts triangle connectivity and coordinates
    for each volume.

    This function provides the same data as pydagmc's volume.get_triangle_conn_and_coords()
    method, returning the triangle mesh data for each volume in the geometry.

    Arguments:
        filename: the filename of the DAGMC h5m file
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A dictionary mapping volume IDs to tuples of (connectivity, coordinates):
        - connectivity: numpy array of shape (n_triangles, 3) containing vertex indices.
          Each row represents a triangle with indices into the coordinates array.
        - coordinates: numpy array of shape (n_vertices, 3) containing 3D vertex
          positions (x, y, z).

    Example:
        >>> import dagmc_h5m_file_inspector as di
        >>> data = di.get_triangle_conn_and_coords_by_volume("dagmc.h5m")
        >>> for vol_id, (connectivity, coords) in data.items():
        ...     print(f"Volume {vol_id}: {len(connectivity)} triangles, {len(coords)} vertices")
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_triangle_conn_and_coords_pymoab(filename)
    else:
        return _get_triangle_conn_and_coords_h5py(filename)
