from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .core import BoundingBox, _load_dagmc_data


class DAGMCFile:
    """A DAGMC h5m file loaded into memory for repeated operations.

    The file is read once during construction. Query methods then use the
    cached geometry and material data without reopening the input file.

    Parameters
    ----------
    filename : str
        Path to the DAGMC h5m file.
    backend : {"h5py", "pymoab"}
        Backend used to read the file.
    """

    def __init__(
        self,
        filename: str,
        backend: Literal["h5py", "pymoab"] = "h5py",
    ) -> None:
        self.filename = filename
        self.backend = backend
        self._data = _load_dagmc_data(filename, backend)

    def get_bounding_box(
        self, materials: Optional[Union[str, List[str]]] = None
    ) -> BoundingBox:
        """Return the bounding box, optionally filtered by material name."""
        if isinstance(materials, str):
            materials = [materials]

        if materials is None:
            volume_ids = self.get_volumes()
        else:
            volume_ids = [
                volume_id
                for volume_id, material in self._data.volume_materials.items()
                if material in materials
            ]
            if not volume_ids:
                available = self.get_materials()
                raise ValueError(
                    f"No volumes found for materials {materials}. "
                    f"Available materials: {available}"
                )

        coordinates = [
            self._data.volume_data[volume_id][1]
            for volume_id in volume_ids
            if self._data.volume_data[volume_id][1].size > 0
        ]
        if not coordinates:
            raise ValueError(f"No triangle data found in {self.filename}")

        all_coordinates = np.concatenate(coordinates)
        return BoundingBox(
            all_coordinates.min(axis=0),
            all_coordinates.max(axis=0),
        )

    def get_materials(self, remove_prefix: bool = True) -> List[str]:
        """Return the sorted material tags in the loaded file."""
        materials = sorted(set(self._data.volume_materials.values()))
        if remove_prefix:
            return materials
        return [f"mat:{material}" for material in materials]

    def get_triangle_conn_and_coords_by_volume(
        self,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return copies of per-volume triangle connectivity and coordinates."""
        return {
            volume_id: (connectivity.copy(), coordinates.copy())
            for volume_id, (connectivity, coordinates) in self._data.volume_data.items()
        }

    def get_volumes(self) -> List[int]:
        """Return the sorted volume IDs in the loaded file."""
        return sorted(self._data.volume_data)

    def get_volumes_and_materials(self, remove_prefix: bool = True) -> Dict[int, str]:
        """Return volume IDs mapped to their material tags."""
        if remove_prefix:
            return dict(self._data.volume_materials)
        return {
            volume_id: f"mat:{material}"
            for volume_id, material in self._data.volume_materials.items()
        }
