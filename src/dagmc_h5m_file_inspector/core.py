from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

RANGE_COMPRESSED_FLAG = 0x8

_VALID_BACKENDS = ("h5py", "pymoab")


def _validate_backend(backend: str) -> None:
    """Raise ValueError if *backend* is not a recognised backend string."""
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend {backend!r}. Must be one of {_VALID_BACKENDS}."
        )


@dataclass(frozen=True)
class _SetInfo:
    """Internal dataclass for storing MOAB set information."""

    handle: int
    contents: Sequence[int] | Sequence[Tuple[int, int]]
    contents_are_ranges: bool
    children: Sequence[int]
    parents: Sequence[int]
    flags: int


# This is a reimplementation of the BoundingBox class that is mainly
# API compatible with OpenMC. One difference is that this one does not
# make any use of Numpy.
# TODO if openmc becomes pip installable then we could make use of
# the openmc BoundingBox directly
class BoundingBox:
    """Axis-aligned bounding box.

    Parameters
    ----------
    lower_left : iterable of float
        Lower-left coordinates of the box (length 3).
    upper_right : iterable of float
        Upper-right coordinates of the box (length 3).
    """

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
        """Intersection of two bounding boxes."""
        if not isinstance(other, BoundingBox):
            return NotImplemented
        new = BoundingBox(self._lower_left, self._upper_right)
        new &= other
        return new

    def __contains__(self, other: object) -> bool:
        """Check if a point or BoundingBox is inside the bounding box."""
        if isinstance(other, BoundingBox):
            return all(
                a <= p <= b
                for p, a, b in zip(
                    other._lower_left, self._lower_left, self._upper_right
                )
            ) and all(
                a <= p <= b
                for p, a, b in zip(
                    other._upper_right, self._lower_left, self._upper_right
                )
            )
        point = tuple(float(v) for v in other)
        if len(point) != 3:
            raise ValueError(f"point must have length 3, got {len(point)}")
        return all(
            a <= p <= b for p, a, b in zip(point, self._lower_left, self._upper_right)
        )

    def __getitem__(self, key) -> Tuple[float, float, float]:
        """Index into the bounding box (0=lower_left, 1=upper_right)."""
        return (self._lower_left, self._upper_right)[key]

    def __iand__(self, other: "BoundingBox") -> "BoundingBox":
        """In-place intersection of two bounding boxes."""
        if not isinstance(other, BoundingBox):
            return NotImplemented
        self._lower_left = tuple(
            max(a, b) for a, b in zip(self._lower_left, other._lower_left)
        )
        self._upper_right = tuple(
            min(a, b) for a, b in zip(self._upper_right, other._upper_right)
        )
        return self

    def __init__(self, lower_left: object, upper_right: object) -> None:
        self._lower_left = tuple(float(v) for v in lower_left)
        self._upper_right = tuple(float(v) for v in upper_right)
        if len(self._lower_left) != 3:
            raise ValueError(
                f"lower_left must have length 3, got {len(self._lower_left)}"
            )
        if len(self._upper_right) != 3:
            raise ValueError(
                f"upper_right must have length 3, got {len(self._upper_right)}"
            )

    def __ior__(self, other: "BoundingBox") -> "BoundingBox":
        """In-place union of two bounding boxes."""
        if not isinstance(other, BoundingBox):
            return NotImplemented
        self._lower_left = tuple(
            min(a, b) for a, b in zip(self._lower_left, other._lower_left)
        )
        self._upper_right = tuple(
            max(a, b) for a, b in zip(self._upper_right, other._upper_right)
        )
        return self

    def __len__(self) -> int:
        """Length of the bounding box (always 2: lower_left + upper_right)."""
        return 2

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        """Union of two bounding boxes."""
        if not isinstance(other, BoundingBox):
            return NotImplemented
        new = BoundingBox(self._lower_left, self._upper_right)
        new |= other
        return new

    def __repr__(self) -> str:
        return f"BoundingBox({self._lower_left}, {self._upper_right})"

    def __setitem__(self, key: int, val: object) -> None:
        """Set lower_left (0) or upper_right (1) by index."""
        val = tuple(float(v) for v in val)
        if key == 0:
            self._lower_left = val
        elif key == 1:
            self._upper_right = val
        else:
            raise IndexError(f"index {key} out of range for BoundingBox")

    @property
    def center(self) -> Tuple[float, float, float]:
        """Center point of the box."""
        return tuple((a + b) / 2.0 for a, b in zip(self._lower_left, self._upper_right))

    def expand(self, padding, inplace=False) -> "BoundingBox":
        """Expand the box by *padding* in all directions.

        Parameters
        ----------
        padding : float or iterable of float
            Amount to expand by. Scalar applies to all axes.
        inplace : bool
            If True, modify this box. Otherwise return a new one.
        """
        try:
            pad = tuple(float(v) for v in padding)
        except TypeError:
            pad = (float(padding),) * 3
        if len(pad) != 3:
            raise ValueError(
                f"padding must be scalar or length 3, got length {len(pad)}"
            )
        ll = tuple(a - p for a, p in zip(self._lower_left, pad))
        ur = tuple(a + p for a, p in zip(self._upper_right, pad))
        if inplace:
            self._lower_left = ll
            self._upper_right = ur
            return self
        return BoundingBox(ll, ur)

    @property
    def extent(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Extent of the box as (left, right, bottom, top) tuples keyed by
        basis plane.  Intended for use with Matplotlib's ``imshow`` *extent*
        parameter."""
        ll, ur = self._lower_left, self._upper_right
        return {
            "xy": (ll[0], ur[0], ll[1], ur[1]),
            "xz": (ll[0], ur[0], ll[2], ur[2]),
            "yz": (ll[1], ur[1], ll[2], ur[2]),
        }

    @classmethod
    def infinite(cls):
        """Return an infinite bounding box."""
        return cls((-float("inf"),) * 3, (float("inf"),) * 3)

    @property
    def lower_left(self) -> Tuple[float, float, float]:
        """Lower-left coordinates of the box."""
        return self._lower_left

    @property
    def upper_right(self) -> Tuple[float, float, float]:
        """Upper-right coordinates of the box."""
        return self._upper_right

    @property
    def volume(self) -> float:
        """Volume of the box (product of widths)."""
        w = self.width
        return w[0] * w[1] * w[2]

    @property
    def width(self) -> Tuple[float, float, float]:
        """Width along each axis (upper_right - lower_left)."""
        return tuple(b - a for a, b in zip(self._lower_left, self._upper_right))


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


def _get_volumes_and_materials_h5py(
    filename: str, remove_prefix: bool
) -> Dict[int, str]:
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


def _get_bounding_box_h5py(filename: str) -> BoundingBox:
    """Get bounding box using h5py backend."""
    with h5py.File(filename, "r") as f:
        coords = f["tstt/nodes/coordinates"][()]
        lower_left = coords.min(axis=0)
        upper_right = coords.max(axis=0)
        return BoundingBox(lower_left, upper_right)


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
                    (int(data[i]), int(data[i + 1])) for i in range(0, len(data), 2)
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
    return {int(h): (int(v[0]), int(v[1])) for h, v in zip(ids, values)}


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

    handles = [h for h in set_info.contents if tri_start <= h <= tri_end]
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


def _surface_area_from_tris(
    coords: np.ndarray,
    tri_conn0: np.ndarray,
    tri_indices: np.ndarray,
) -> float:
    """Calculate total surface area from triangles using cross product."""
    tri_nodes = tri_conn0[tri_indices]
    v0 = coords[tri_nodes[:, 0]]
    v1 = coords[tri_nodes[:, 1]]
    v2 = coords[tri_nodes[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    return float(0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1).sum())


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
            h
            for h in surface_handles
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


def _surface_areas_for_volume_set(
    *,
    vol_handle: int,
    sets_by_handle: Dict[int, _SetInfo],
    surface_handles: set,
    geom_sense: Dict[int, Tuple[int, int]],
    coords: np.ndarray,
    tri_conn0: np.ndarray,
    tri_start: int,
    tri_end: int,
) -> List[float]:
    """Calculate individual surface areas for each DAGMC surface bounding a volume."""
    volume_set = sets_by_handle.get(vol_handle)
    if volume_set is None:
        return []

    if volume_set.children:
        surfaces = [h for h in volume_set.children if h in surface_handles]
    else:
        surfaces = [
            h
            for h in surface_handles
            if h in geom_sense and vol_handle in geom_sense[h]
        ]

    areas: List[float] = []
    for surf_handle in surfaces:
        surf_set = sets_by_handle.get(surf_handle)
        if surf_set is None:
            continue

        tri_indices = _tri_indices_for_set(
            surf_set,
            tri_start=tri_start,
            tri_end=tri_end,
        )
        if tri_indices.size == 0:
            continue

        areas.append(_surface_area_from_tris(coords, tri_conn0, tri_indices))

    return areas


def _get_surface_areas_h5py(filename: str) -> Dict[int, List[float]]:
    """Get surface areas for each volume ID using h5py backend.

    Returns a dictionary mapping volume IDs to lists of surface areas,
    one entry per DAGMC surface bounding the volume.
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

        global_ids: Dict[int, int] = {}
        sets_start_id = int(f["tstt/sets/list"].attrs["start_id"])
        if "tstt/sets/tags/GLOBAL_ID" in f:
            dense_gids = f["tstt/sets/tags/GLOBAL_ID"][...]
            for idx, gid in enumerate(dense_gids):
                handle = sets_start_id + idx
                global_ids[handle] = int(gid)
        else:
            global_ids = _read_tag_h5py(f, "GLOBAL_ID")

        surface_handles = {h for h, cat in categories.items() if cat == "Surface"}
        surface_handles.update(h for h, dim in geom_dim.items() if dim == 2)

        volume_handles = {h for h, cat in categories.items() if cat == "Volume"}
        volume_handles.update(h for h, dim in geom_dim.items() if dim == 3)

        result: Dict[int, List[float]] = {}
        for vol_handle in volume_handles:
            vol_gid = global_ids.get(vol_handle)
            if vol_gid is None:
                continue

            areas = _surface_areas_for_volume_set(
                vol_handle=vol_handle,
                sets_by_handle=sets_by_handle,
                surface_handles=surface_handles,
                geom_sense=geom_sense,
                coords=coords,
                tri_conn0=tri_conn0,
                tri_start=tri_start,
                tri_end=tri_end,
            )
            result[int(vol_gid)] = areas

        return result


def _get_volumes_sizes_h5py(filename: str) -> Dict[int, float]:
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
        surface_handles = {h for h, cat in categories.items() if cat == "Surface"}
        surface_handles.update(h for h, dim in geom_dim.items() if dim == 2)

        # Build set of volume handles
        volume_handles = {h for h, cat in categories.items() if cat == "Volume"}
        volume_handles.update(h for h, dim in geom_dim.items() if dim == 3)

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


def _check_pymoab_available() -> None:
    """Check if pymoab is available and raise ImportError if not."""
    try:
        import pymoab  # noqa: F401
    except ImportError:
        raise ImportError(
            "pymoab is not installed. Install it to use backend='pymoab', "
            "or use the default h5py backend."
        )


def _load_moab_file(filename: str) -> object:
    """Load a DAGMC h5m file into a pymoab Core object."""
    from pymoab import core

    moab_core = core.Core()
    moab_core.load_file(filename)
    return moab_core


def _get_groups_pymoab(mbcore: object) -> object:
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


def _get_volumes_and_materials_pymoab(
    filename: str, remove_prefix: bool
) -> Dict[int, str]:
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


def _get_bounding_box_pymoab(filename: str) -> BoundingBox:
    """Get bounding box using pymoab backend."""
    import pymoab as mb

    mbcore = _load_moab_file(filename)
    # Get all vertices
    vertices = mbcore.get_entities_by_type(0, mb.types.MBVERTEX)
    coords = mbcore.get_coords(vertices)
    coords = coords.reshape(-1, 3)

    lower_left = coords.min(axis=0)
    upper_right = coords.max(axis=0)
    return BoundingBox(lower_left, upper_right)


def _get_triangle_conn_and_coords_h5py(
    filename: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
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
        surface_handles = {h for h, cat in categories.items() if cat == "Surface"}
        surface_handles.update(h for h, dim in geom_dim.items() if dim == 2)

        # Build set of volume handles
        volume_handles = {h for h, cat in categories.items() if cat == "Volume"}
        volume_handles.update(h for h, dim in geom_dim.items() if dim == 3)

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


def _write_h5m(
    filename: str,
    volumes_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    vol_mat_mapping: Dict[int, str],
) -> None:
    """Write a DAGMC h5m file from per-volume triangle data using h5py.

    Arguments:
        filename: output file path
        volumes_data: dict mapping volume_id -> (connectivity, coordinates)
            where connectivity is Mx3 (0-based into coordinates) and
            coordinates is Nx3 float64
        vol_mat_mapping: dict mapping volume_id -> material_name (without
            ``mat:`` prefix)
    """
    from datetime import datetime

    # Sort volume IDs for deterministic output
    vol_ids = sorted(volumes_data.keys())

    # Merge per-volume coordinates into a global vertex array and
    # adjust connectivity to global indices.
    all_coords: List[np.ndarray] = []
    all_triangles: List[np.ndarray] = []
    # Track per-volume triangle ranges (start_idx, count) in global array
    vol_tri_ranges: Dict[int, Tuple[int, int]] = {}
    # Track per-volume vertex ranges (start_idx, count) in global array
    vol_vert_ranges: Dict[int, Tuple[int, int]] = {}
    vert_offset = 0
    tri_offset = 0
    for vid in vol_ids:
        conn, coords = volumes_data[vid]
        n_verts = len(coords)
        n_tris = len(conn)
        if n_tris == 0:
            vol_tri_ranges[vid] = (tri_offset, 0)
            vol_vert_ranges[vid] = (vert_offset, 0)
            continue
        all_coords.append(coords)
        all_triangles.append(conn + vert_offset)
        vol_vert_ranges[vid] = (vert_offset, n_verts)
        vol_tri_ranges[vid] = (tri_offset, n_tris)
        vert_offset += n_verts
        tri_offset += n_tris

    if not all_coords:
        vertices_arr = np.empty((0, 3), dtype=np.float64)
        triangles_arr = np.empty((0, 3), dtype=np.int64)
    else:
        vertices_arr = np.concatenate(all_coords, axis=0)
        triangles_arr = np.concatenate(all_triangles, axis=0)

    num_vertices = len(vertices_arr)
    num_triangles = len(triangles_arr)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, "w") as f:
        tstt = f.create_group("tstt")
        global_id = 1

        # === NODES ===
        nodes_group = tstt.create_group("nodes")
        coords_ds = nodes_group.create_dataset("coordinates", data=vertices_arr)
        coords_ds.attrs.create("start_id", global_id)
        global_id += num_vertices

        node_tags = nodes_group.create_group("tags")
        node_tags.create_dataset(
            "GLOBAL_ID", data=np.full(num_vertices, -1, dtype=np.int32)
        )

        # === ELEMENTS ===
        elements = tstt.create_group("elements")
        elem_enum = {"Tri": 2}
        tstt["elemtypes"] = h5py.enum_dtype(elem_enum)

        now = datetime.now()
        tstt.create_dataset(
            "history",
            data=[
                "dagmc_h5m_file_inspector".encode("ascii"),
                now.strftime("%m/%d/%y").encode("ascii"),
                now.strftime("%H:%M:%S").encode("ascii"),
            ],
        )

        tri3_group = elements.create_group("Tri3")
        tri3_group.attrs.create(
            "element_type", elem_enum["Tri"], dtype=tstt["elemtypes"]
        )
        connectivity_ds = tri3_group.create_dataset(
            "connectivity",
            data=triangles_arr + 1,  # 1-based vertex IDs in h5m
            dtype=np.uint64,
        )
        triangle_start_id = global_id
        connectivity_ds.attrs.create("start_id", triangle_start_id)
        global_id += num_triangles

        tags_tri3 = tri3_group.create_group("tags")
        tags_tri3.create_dataset(
            "GLOBAL_ID", data=np.full(num_triangles, -1, dtype=np.int32)
        )

        # === SETS layout ===
        # File set first (position 0) to absorb the off-by-one in the
        # h5py reader's entity-handle mapping, then surfaces, volumes,
        # groups — matching cad_to_dagmc's structure.
        sets_start_id = global_id

        surface_set_ids: Dict[int, int] = {}
        volume_set_ids: Dict[int, int] = {}
        current_set_id = sets_start_id

        # File set (position 0 — NOT tagged with CATEGORY)
        file_set_id = current_set_id
        current_set_id += 1

        # Surface sets (one per volume)
        for vid in vol_ids:
            surface_set_ids[vid] = current_set_id
            current_set_id += 1

        # Volume sets
        for vid in vol_ids:
            volume_set_ids[vid] = current_set_id
            current_set_id += 1

        # Group sets (one per unique material)
        unique_materials = sorted(set(vol_mat_mapping[v] for v in vol_ids))
        mat_to_group_set_id: Dict[str, int] = {}
        for mat_name in unique_materials:
            mat_to_group_set_id[mat_name] = current_set_id
            current_set_id += 1

        global_id = current_set_id

        # === TAGS ===
        tstt_tags = tstt.create_group("tags")

        # CATEGORY
        category_set_ids_list: List[int] = []
        categories: List[str] = []
        geom_dim_set_ids_list: List[int] = []
        geom_dimensions: List[int] = []

        for vid in vol_ids:
            category_set_ids_list.append(volume_set_ids[vid])
            categories.append("Volume")
            geom_dim_set_ids_list.append(volume_set_ids[vid])
            geom_dimensions.append(3)

        for mat_name in unique_materials:
            category_set_ids_list.append(mat_to_group_set_id[mat_name])
            categories.append("Group")

        for vid in vol_ids:
            category_set_ids_list.append(surface_set_ids[vid])
            categories.append("Surface")
            geom_dim_set_ids_list.append(surface_set_ids[vid])
            geom_dimensions.append(2)

        cat_group = tstt_tags.create_group("CATEGORY")
        cat_group.attrs.create("class", 1, dtype=np.int32)
        cat_group.create_dataset(
            "id_list", data=np.array(category_set_ids_list, dtype=np.uint64)
        )
        opaque_dt = h5py.opaque_dtype(np.dtype("V32"))
        cat_group["type"] = opaque_dt
        cat_values = np.array(
            [s.encode("ascii").ljust(32, b"\x00") for s in categories], dtype="V32"
        )
        cat_group.create_dataset("values", data=cat_values)

        # GEOM_DIMENSION
        geom_group = tstt_tags.create_group("GEOM_DIMENSION")
        geom_group["type"] = np.dtype("i4")
        geom_group.attrs.create("class", 1, dtype=np.int32)
        geom_group.attrs.create("default", -1, dtype=geom_group["type"])
        geom_group.attrs.create("global", -1, dtype=geom_group["type"])
        geom_group.create_dataset(
            "id_list", data=np.array(geom_dim_set_ids_list, dtype=np.uint64)
        )
        geom_group.create_dataset(
            "values", data=np.array(geom_dimensions, dtype=np.int32)
        )

        # GEOM_SENSE_2
        surface_ids_for_sense = [surface_set_ids[vid] for vid in vol_ids]
        gs2_group = tstt_tags.create_group("GEOM_SENSE_2")
        gs2_dtype = np.dtype("(2,)u8")
        gs2_group["type"] = gs2_dtype
        gs2_group.attrs.create("class", 1, dtype=np.int32)
        gs2_group.attrs.create("is_handle", 1, dtype=np.int32)
        gs2_group.create_dataset(
            "id_list", data=np.array(surface_ids_for_sense, dtype=np.uint64)
        )

        sense_values = []
        for vid in vol_ids:
            vol = volume_set_ids[vid]
            sense_values.append([vol, 0])

        if sense_values:
            gs2_values = np.zeros((len(sense_values),), dtype=[("f0", "<u8", (2,))])
            gs2_values["f0"] = np.array(sense_values, dtype=np.uint64)
            gs2_space = h5py.h5s.create_simple((len(sense_values),))
            gs2_arr_type = h5py.h5t.array_create(h5py.h5t.NATIVE_UINT64, (2,))
            gs2_dset = h5py.h5d.create(gs2_group.id, b"values", gs2_arr_type, gs2_space)
            gs2_dset.write(h5py.h5s.ALL, h5py.h5s.ALL, gs2_values, mtype=gs2_arr_type)
            gs2_dset.close()

        # GLOBAL_ID (sparse tag)
        gid_ids: List[int] = []
        gid_values: List[int] = []
        for vid in vol_ids:
            gid_ids.append(surface_set_ids[vid])
            gid_values.append(vid)
        for vid in vol_ids:
            gid_ids.append(volume_set_ids[vid])
            gid_values.append(vid)
        for mat_name in unique_materials:
            gid_ids.append(mat_to_group_set_id[mat_name])
            gid_values.append(-1)

        gid_group = tstt_tags.create_group("GLOBAL_ID")
        gid_group["type"] = np.dtype("i4")
        gid_group.attrs.create("class", 2, dtype=np.int32)
        gid_group.attrs.create("default", -1, dtype=gid_group["type"])
        gid_group.attrs.create("global", -1, dtype=gid_group["type"])
        gid_group.create_dataset("id_list", data=np.array(gid_ids, dtype=np.uint64))
        gid_group.create_dataset("values", data=np.array(gid_values, dtype=np.int32))

        # NAME tag (for groups)
        name_ids: List[int] = []
        name_values: List[str] = []
        for mat_name in unique_materials:
            name_ids.append(mat_to_group_set_id[mat_name])
            name_values.append(f"mat:{mat_name}")

        name_group = tstt_tags.create_group("NAME")
        name_group.attrs.create("class", 1, dtype=np.int32)
        name_group.create_dataset("id_list", data=np.array(name_ids, dtype=np.uint64))
        name_group["type"] = h5py.opaque_dtype(np.dtype("S32"))
        name_group.create_dataset("values", data=name_values, dtype=name_group["type"])

        for tag_name in ["DIRICHLET_SET", "MATERIAL_SET", "NEUMANN_SET"]:
            tag_grp = tstt_tags.create_group(tag_name)
            tag_grp["type"] = np.dtype("i4")
            tag_grp.attrs.create("class", 1, dtype=np.int32)
            tag_grp.attrs.create("default", -1, dtype=tag_grp["type"])
            tag_grp.attrs.create("global", -1, dtype=tag_grp["type"])

        # === SETS structure ===
        sets_group = tstt.create_group("sets")

        contents_arr: List[int] = []
        list_rows: List[List[int]] = []
        parents_list: List[int] = []
        children_list: List[int] = []

        contents_end = -1
        children_end = -1
        parents_end = -1

        # File set first (position 0) — contains everything
        last_handle = max(
            [file_set_id]
            + list(surface_set_ids.values())
            + list(volume_set_ids.values())
            + list(mat_to_group_set_id.values())
        )
        if num_vertices > 0 or num_triangles > 0:
            contents_arr.extend([1, last_handle])
            contents_end = len(contents_arr) - 1
        list_rows.append([contents_end, children_end, parents_end, 10])

        # Surface sets — each contains the vertices + triangles for one volume
        for vid in vol_ids:
            tri_start_idx, tri_count = vol_tri_ranges[vid]
            vert_start_idx, vert_count = vol_vert_ranges[vid]

            # Vertex handles (1-based)
            for i in range(vert_count):
                contents_arr.append(vert_start_idx + i + 1)

            # Triangle handles
            for i in range(tri_count):
                contents_arr.append(triangle_start_id + tri_start_idx + i)

            contents_end = len(contents_arr) - 1

            # Parent = the volume set
            parents_list.append(volume_set_ids[vid])
            parents_end = len(parents_list) - 1

            # flags: 2 = MESHSET_SET
            list_rows.append([contents_end, children_end, parents_end, 2])

        # Volume sets — have surface as child, no direct content
        for vid in vol_ids:
            children_list.append(surface_set_ids[vid])
            children_end = len(children_list) - 1
            list_rows.append([contents_end, children_end, parents_end, 2])

        # Group sets — contain volume handles
        for mat_name in unique_materials:
            vols_in_mat = [vid for vid in vol_ids if vol_mat_mapping[vid] == mat_name]
            for vid in vols_in_mat:
                contents_arr.append(volume_set_ids[vid])
            contents_end = len(contents_arr) - 1
            list_rows.append([contents_end, children_end, parents_end, 2])

        sets_group.create_dataset(
            "contents", data=np.array(contents_arr, dtype=np.uint64)
        )
        sets_group.create_dataset(
            "children",
            data=np.array(children_list, dtype=np.uint64)
            if children_list
            else np.array([], dtype=np.uint64),
        )
        sets_group.create_dataset(
            "parents",
            data=np.array(parents_list, dtype=np.uint64)
            if parents_list
            else np.array([], dtype=np.uint64),
        )

        lst = sets_group.create_dataset(
            "list", data=np.array(list_rows, dtype=np.int64)
        )
        lst.attrs.create("start_id", sets_start_id)

        # Dense GLOBAL_ID for sets (matches set list order: file, surfaces,
        # volumes, groups)
        set_global_ids: List[int] = []
        set_global_ids.append(-1)  # file set (position 0)
        for vid in vol_ids:
            set_global_ids.append(vid)  # surface
        for vid in vol_ids:
            set_global_ids.append(vid)  # volume
        for _mat_name in unique_materials:
            set_global_ids.append(-1)  # group

        sets_tags = sets_group.create_group("tags")
        sets_tags.create_dataset(
            "GLOBAL_ID", data=np.array(set_global_ids, dtype=np.int32)
        )

        tstt.attrs.create("max_id", np.uint64(global_id - 1))


def _remove_materials_h5py(
    input_filename: str,
    output_filename: str,
    materials_to_remove: List[str],
) -> List[str]:
    """Remove materials using h5py backend (read-filter-write approach)."""
    vol_mat = get_volumes_and_materials_from_h5m(
        filename=input_filename, remove_prefix=True, backend="h5py"
    )
    all_materials = sorted(set(vol_mat.values()))
    matched = sorted(set(materials_to_remove) & set(all_materials))
    if not matched:
        raise ValueError(
            f"None of the specified materials {materials_to_remove} found in "
            f"{input_filename}. Available materials: {all_materials}"
        )

    vol_data = get_triangle_conn_and_coords_by_volume(
        filename=input_filename, backend="h5py"
    )

    keep_vols = {
        vid: mat for vid, mat in vol_mat.items() if mat not in materials_to_remove
    }

    if not keep_vols:
        # All volumes removed — write an empty-ish file
        _write_h5m(output_filename, {}, {})
    else:
        keep_data = {vid: vol_data[vid] for vid in keep_vols}
        _write_h5m(output_filename, keep_data, keep_vols)

    return matched


def _remove_materials_pymoab(
    input_filename: str,
    output_filename: str,
    materials_to_remove: List[str],
) -> List[str]:
    """Remove materials using pymoab backend.

    Uses the same read-filter-write approach as the h5py backend: reads the
    data, filters out unwanted volumes, and writes a fresh file using
    ``_write_h5m``.  This avoids issues with pymoab's ``write_file`` when
    all groups are removed.
    """
    vol_mat = get_volumes_and_materials_from_h5m(
        filename=input_filename, remove_prefix=True, backend="pymoab"
    )
    all_materials = sorted(set(vol_mat.values()))
    matched = sorted(set(materials_to_remove) & set(all_materials))
    if not matched:
        raise ValueError(
            f"None of the specified materials {materials_to_remove} found in "
            f"{input_filename}. Available materials: {all_materials}"
        )

    vol_data = get_triangle_conn_and_coords_by_volume(
        filename=input_filename, backend="pymoab"
    )

    keep_vols = {
        vid: mat for vid, mat in vol_mat.items() if mat not in materials_to_remove
    }

    if not keep_vols:
        _write_h5m(output_filename, {}, {})
    else:
        keep_data = {vid: vol_data[vid] for vid in keep_vols}
        _write_h5m(output_filename, keep_data, keep_vols)

    return matched


def _get_triangle_conn_and_coords_pymoab(
    filename: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Get triangle connectivity and coordinates for each volume using pymoab backend.

    Returns a dictionary mapping volume IDs to tuples of (connectivity, coordinates)
    where connectivity is an Mx3 array of vertex indices and coordinates is an Nx3
    array of 3D points. The connectivity indices are 0-based relative to the
    coordinates array for that volume.
    """
    import pymoab as mb

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


def _get_volumes_sizes_pymoab(filename: str) -> Dict[int, float]:
    """Get geometric volume sizes for each volume ID using pymoab backend.

    Uses GEOM_SENSE_2 tag to determine surface orientation relative to each
    volume, enabling correct signed volume calculation for nested geometries.
    """
    import pymoab as mb

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


def _get_surface_areas_pymoab(filename: str) -> Dict[int, List[float]]:
    """Get surface areas for each volume ID using pymoab backend.

    Returns a dictionary mapping volume IDs to lists of surface areas,
    one entry per DAGMC surface bounding the volume.
    """
    import pymoab as mb

    mbcore = _load_moab_file(filename)
    category_tag = mbcore.tag_get_handle(mb.types.CATEGORY_TAG_NAME)
    id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)

    volume_ents = mbcore.get_entities_by_type_and_tag(
        0, mb.types.MBENTITYSET, category_tag, ["Volume"]
    )

    result: Dict[int, List[float]] = {}

    for vol_ent in volume_ents:
        vol_gid = mbcore.tag_get_data(id_tag, vol_ent)[0][0].item()
        surfaces = mbcore.get_child_meshsets(vol_ent)

        areas: List[float] = []
        for surf in surfaces:
            tris = mbcore.get_entities_by_type(surf, mb.types.MBTRI)
            if not tris:
                continue

            all_verts = set()
            for tri in tris:
                conn = mbcore.get_connectivity(tri)
                all_verts.update(conn)

            all_verts = list(all_verts)
            vert_to_idx = {v: i for i, v in enumerate(all_verts)}

            coords = mbcore.get_coords(all_verts).reshape(-1, 3)

            tri_array = []
            for tri in tris:
                conn = mbcore.get_connectivity(tri)
                tri_array.append([vert_to_idx[v] for v in conn])
            tri_array = np.array(tri_array)

            v0 = coords[tri_array[:, 0]]
            v1 = coords[tri_array[:, 1]]
            v2 = coords[tri_array[:, 2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = float(0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1).sum())
            areas.append(area)

        result[vol_gid] = areas

    return result


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
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_volumes_pymoab(filename)
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
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_materials_pymoab(filename, remove_prefix)
    return _get_materials_h5py(filename, remove_prefix)


def get_volumes_and_materials_from_h5m(
    filename: str,
    remove_prefix: Optional[bool] = True,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> Dict[int, str]:
    """Reads in a DAGMC h5m file and finds the volume ids with their
    associated material tags.

    Arguments:
        filename: the filename of the DAGMC h5m file
        remove_prefix: remove the mat: prefix from the material tag or not
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A dictionary of volume ids and material tags
    """
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_volumes_and_materials_pymoab(filename, remove_prefix)
    return _get_volumes_and_materials_h5py(filename, remove_prefix)


def get_bounding_box_from_h5m(
    filename: str,
    materials: Optional[Union[str, List[str]]] = None,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> "BoundingBox":
    """Reads in a DAGMC h5m file and returns the axis-aligned bounding box.

    Arguments:
        filename: the filename of the DAGMC h5m file
        materials: optional material name or list of material names to filter
            the bounding box by. If None, the bounding box of all geometry is
            returned. If a string, the bounding box of that single material is
            returned. If a list of strings, the combined bounding box of all
            specified materials is returned.
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A BoundingBox object representing the axis-aligned bounding box.
    """
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if materials is None:
        if backend == "pymoab":
            _check_pymoab_available()
            return _get_bounding_box_pymoab(filename)
        return _get_bounding_box_h5py(filename)

    if isinstance(materials, str):
        materials = [materials]

    vol_mat_mapping = get_volumes_and_materials_from_h5m(
        filename=filename,
        remove_prefix=True,
        backend=backend,
    )

    matching_vol_ids = [
        vol_id for vol_id, mat_name in vol_mat_mapping.items() if mat_name in materials
    ]

    if not matching_vol_ids:
        available = sorted(set(vol_mat_mapping.values()))
        raise ValueError(
            f"No volumes found for materials {materials}. "
            f"Available materials: {available}"
        )

    vol_data = get_triangle_conn_and_coords_by_volume(
        filename=filename,
        backend=backend,
    )

    all_coords = np.concatenate(
        [
            vol_data[vol_id][1]
            for vol_id in matching_vol_ids
            if vol_id in vol_data and vol_data[vol_id][1].size > 0
        ]
    )

    lower_left = all_coords.min(axis=0)
    upper_right = all_coords.max(axis=0)
    return BoundingBox(lower_left, upper_right)


def get_volumes_from_h5m_by_cell_id(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> Dict[int, float]:
    """Reads in a DAGMC h5m file and calculates the geometric volume
    (size) of each volume entity.

    Arguments:
        filename: the filename of the DAGMC h5m file
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A dictionary mapping volume IDs (cell IDs) to their geometric volumes (sizes)
    """
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_volumes_sizes_pymoab(filename)
    return _get_volumes_sizes_h5py(filename)


def get_volumes_from_h5m_by_material_name(
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
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    # Get volume-to-material mapping and volume sizes
    vol_mat_mapping = get_volumes_and_materials_from_h5m(
        filename=filename,
        remove_prefix=True,
        backend=backend,
    )
    volume_sizes = get_volumes_from_h5m_by_cell_id(
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


def get_volumes_from_h5m_by_cell_id_and_material_name(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> Dict[Tuple[int, str], float]:
    """Reads in a DAGMC h5m file and calculates the geometric volume
    for each cell, returning results keyed by both cell ID and material name.

    Arguments:
        filename: the filename of the DAGMC h5m file
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A dictionary mapping (cell_id, material_name) tuples to their geometric volumes.
    """
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    # Get volume-to-material mapping and volume sizes
    vol_mat_mapping = get_volumes_and_materials_from_h5m(
        filename=filename,
        remove_prefix=True,
        backend=backend,
    )
    volume_sizes = get_volumes_from_h5m_by_cell_id(
        filename=filename,
        backend=backend,
    )

    # Build dictionary with (cell_id, material_name) tuple keys
    result: Dict[Tuple[int, str], float] = {}
    for vol_id, mat_name in vol_mat_mapping.items():
        result[(vol_id, mat_name)] = volume_sizes.get(vol_id, 0.0)

    return result


def get_surface_area_by_cell_id(
    filename: str,
    cell_id: int,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> List[float]:
    """Returns the surface area of each DAGMC surface bounding the given volume.

    Arguments:
        filename: the filename of the DAGMC h5m file
        cell_id: the DAGMC volume (cell) ID
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A list of surface areas, one per DAGMC surface bounding the volume.
    """
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        all_areas = _get_surface_areas_pymoab(filename)
    else:
        all_areas = _get_surface_areas_h5py(filename)

    if cell_id not in all_areas:
        raise ValueError(
            f"cell_id {cell_id} not found. "
            f"Available cell IDs: {sorted(all_areas.keys())}"
        )
    return all_areas[cell_id]


def get_surface_area_by_material_name(
    filename: str,
    material: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> List[float]:
    """Returns surface areas for surfaces bounding volumes with the given material.

    Arguments:
        filename: the filename of the DAGMC h5m file
        material: the material tag name (without 'mat:' prefix)
        backend: the backend to use for reading the file ("h5py" or "pymoab")

    Returns:
        A list of surface areas for all DAGMC surfaces bounding volumes
        with the given material name.
    """
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    vol_mat_mapping = get_volumes_and_materials_from_h5m(
        filename=filename,
        remove_prefix=True,
        backend=backend,
    )

    matching_vol_ids = [
        vol_id for vol_id, mat_name in vol_mat_mapping.items() if mat_name == material
    ]

    if not matching_vol_ids:
        available = sorted(set(vol_mat_mapping.values()))
        raise ValueError(
            f"No volumes found for material {material!r}. "
            f"Available materials: {available}"
        )

    if backend == "pymoab":
        _check_pymoab_available()
        all_areas = _get_surface_areas_pymoab(filename)
    else:
        all_areas = _get_surface_areas_h5py(filename)

    combined: List[float] = []
    for vol_id in matching_vol_ids:
        combined.extend(all_areas.get(vol_id, []))

    return combined


def set_openmc_material_volumes_from_h5m(
    materials: Union[List, object],
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
    _validate_backend(backend)
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
    material_volumes = get_volumes_from_h5m_by_material_name(
        filename=filename,
        backend=backend,
    )

    # Set volumes on matching OpenMC materials
    for mat in materials:
        if mat.name is not None and mat.name in material_volumes:
            mat.volume = material_volumes[mat.name]


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
    _validate_backend(backend)
    if not Path(h5m_filename).is_file():
        raise FileNotFoundError(f"filename provided ({h5m_filename}) does not exist")

    if vtkhdf_filename == "":
        vtkhdf_filename = str(Path(h5m_filename).with_suffix(".vtkhdf"))

    # Get per-volume triangle data
    per_volume_data = get_triangle_conn_and_coords_by_volume(
        filename=h5m_filename, backend=backend
    )

    # Get volume-to-material mapping
    vol_mat = get_volumes_and_materials_from_h5m(
        filename=h5m_filename, remove_prefix=True, backend=backend
    )

    # Build unique material name -> integer index mapping
    unique_materials = sorted(set(vol_mat.values()))
    mat_to_int = {name: idx for idx, name in enumerate(unique_materials)}

    # Merge per-volume meshes into global arrays
    all_points = []
    all_conn = []
    all_cell_ids = []
    all_material_ids = []

    point_offset = 0
    for vol_id in sorted(per_volume_data.keys()):
        conn, coords = per_volume_data[vol_id]
        if len(conn) == 0:
            continue
        all_points.append(coords)
        all_conn.append(conn + point_offset)
        n_tris = len(conn)
        all_cell_ids.extend([vol_id] * n_tris)
        mat_name = vol_mat.get(vol_id, "")
        all_material_ids.extend([mat_to_int.get(mat_name, -1)] * n_tris)
        point_offset += len(coords)

    if not all_points:
        raise ValueError(f"No triangle data found in {h5m_filename}")

    global_points = np.concatenate(all_points, axis=0)
    global_conn = np.concatenate(all_conn, axis=0)

    _write_vtkhdf(
        filename=vtkhdf_filename,
        points=global_points,
        connectivity=global_conn,
        cell_ids=np.array(all_cell_ids, dtype=np.int32),
        material_ids=np.array(all_material_ids, dtype=np.int32),
        material_names=unique_materials,
    )

    return vtkhdf_filename


def _write_vtkhdf(
    filename: str,
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_ids: np.ndarray,
    material_ids: np.ndarray,
    material_names: List[str],
) -> None:
    """Write triangle mesh data to a VTKHDF UnstructuredGrid file.

    Arguments:
        filename: path for the output file
        points: vertex coordinates, shape (n_points, 3)
        connectivity: triangle vertex indices, shape (n_triangles, 3), 0-based
        cell_ids: DAGMC volume ID per triangle, shape (n_triangles,)
        material_ids: integer material index per triangle, shape (n_triangles,)
        material_names: list mapping material_id index to material name string
    """
    n_points = points.shape[0]
    n_cells = connectivity.shape[0]
    n_connectivity_ids = n_cells * 3

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 1], dtype=np.int64)
        ascii_type = "UnstructuredGrid".encode("ascii")
        root.attrs.create(
            "Type",
            ascii_type,
            dtype=h5py.string_dtype("ascii", len(ascii_type)),
        )

        root.create_dataset("NumberOfPoints", data=np.array([n_points], dtype=np.int64))
        root.create_dataset("NumberOfCells", data=np.array([n_cells], dtype=np.int64))
        root.create_dataset(
            "NumberOfConnectivityIds",
            data=np.array([n_connectivity_ids], dtype=np.int64),
        )

        root.create_dataset("Points", data=points.astype(np.float64))
        root.create_dataset(
            "Connectivity", data=connectivity.flatten().astype(np.int64)
        )

        offsets = np.arange(0, n_cells * 3 + 1, 3, dtype=np.int64)
        root.create_dataset("Offsets", data=offsets)

        VTK_TRIANGLE = 5
        root.create_dataset(
            "Types", data=np.full(n_cells, VTK_TRIANGLE, dtype=np.uint8)
        )

        cell_data = root.create_group("CellData")
        cell_data.create_dataset("cell_id", data=cell_ids.astype(np.int32))
        cell_data.create_dataset("material_id", data=material_ids.astype(np.int32))

        field_data = root.create_group("FieldData")
        dt = h5py.string_dtype()
        field_data.create_dataset("material_names", data=material_names, dtype=dt)


def get_triangle_conn_and_coords_by_volume(
    filename: str,
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Reads a DAGMC h5m file and extracts triangle connectivity and coordinates
    for each volume.

    This function provides the same data as pydagmc's
    ``volume.get_triangle_conn_and_coords()`` method, returning the
    triangle mesh data for each volume in the geometry.

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
        ...     n_tri, n_vert = len(connectivity), len(coords)
        ...     print(f"Volume {vol_id}: {n_tri} triangles, {n_vert} vertices")
    """
    _validate_backend(backend)
    if not Path(filename).is_file():
        raise FileNotFoundError(f"filename provided ({filename}) does not exist")

    if backend == "pymoab":
        _check_pymoab_available()
        return _get_triangle_conn_and_coords_pymoab(filename)
    return _get_triangle_conn_and_coords_h5py(filename)


def remove_materials_from_h5m(
    input_filename: str,
    output_filename: str,
    materials_to_remove: Union[str, List[str]],
    backend: Literal["h5py", "pymoab"] = "h5py",
) -> List[str]:
    """Remove materials from a DAGMC h5m file and write a new file without them.

    Volumes belonging to the removed materials are excluded from the output.

    Arguments:
        input_filename: path to the input DAGMC h5m file
        output_filename: path for the output h5m file
        materials_to_remove: material name or list of material names to remove
            (without the ``mat:`` prefix, matching the convention of
            ``get_materials_from_h5m(remove_prefix=True)``)
        backend: the backend to use ("h5py" or "pymoab")

    Returns:
        A sorted list of material names that were actually removed.

    Raises:
        FileNotFoundError: If *input_filename* does not exist.
        ValueError: If none of the specified materials are found in the file.
    """
    _validate_backend(backend)
    if not Path(input_filename).is_file():
        raise FileNotFoundError(f"filename provided ({input_filename}) does not exist")

    if isinstance(materials_to_remove, str):
        materials_to_remove = [materials_to_remove]

    if backend == "pymoab":
        _check_pymoab_available()
        return _remove_materials_pymoab(
            input_filename, output_filename, materials_to_remove
        )
    return _remove_materials_h5py(input_filename, output_filename, materials_to_remove)
