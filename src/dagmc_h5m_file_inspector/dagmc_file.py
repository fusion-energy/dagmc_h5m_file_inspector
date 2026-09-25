from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from . import core
from .core import BoundingBox, _load_dagmc_data, _rotation_matrix, _write_h5m


class DAGMCFile:
    """A DAGMC h5m file loaded into memory for repeated operations.

    Geometry and material data are read once during construction and reused by
    in-memory query and mutation methods. Methods for additional source-file
    metadata use the configured filename and backend, and therefore cannot be
    called after an in-memory mutation until the result is written and reloaded.

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
        self._modified = False

    def _require_unmodified_source(self, method_name: str) -> None:
        """Reject source-file operations after in-memory data has changed."""
        if self._modified:
            raise RuntimeError(
                f"{method_name} reads source-file metadata that is unavailable after "
                "an in-memory mutation. Write the file and load the output in a new "
                "DAGMCFile instance first."
            )

    @classmethod
    def combine_h5m_files(
        cls,
        input_files: List[str],
        output_file: str = "dagmc_combined.h5m",
        backend: Literal["h5py", "pymoab"] = "h5py",
    ) -> "DAGMCFile":
        """Combine files and return the loaded combined DAGMC file."""
        core.combine_h5m_files(input_files, output_file, backend=backend)
        return cls(output_file, backend=backend)

    def convert_to_vtkhdf(self, vtkhdf_filename: str = "") -> str:
        """Convert the source DAGMC file to VTKHDF for visualization."""
        self._require_unmodified_source("convert_to_vtkhdf")
        return core.convert_h5m_to_vtkhdf(
            self.filename,
            vtkhdf_filename=vtkhdf_filename,
            backend=self.backend,
        )

    def get_cell_ids_by_group_name(self) -> Dict[str, List[int]]:
        """Return non-material group names mapped to their cell IDs."""
        self._require_unmodified_source("get_cell_ids_by_group_name")
        return core.get_cell_ids_by_group_name(self.filename, backend=self.backend)

    def get_groups_by_cell_id(self) -> Dict[int, List[str]]:
        """Return cell IDs mapped to their non-material group names."""
        self._require_unmodified_source("get_groups_by_cell_id")
        return core.get_groups_by_cell_id(self.filename, backend=self.backend)

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
            if volume_id in self._data.volume_data
            and self._data.volume_data[volume_id][1].size > 0
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
        materials = sorted(set(self._data.materials))
        if remove_prefix:
            return materials
        return [f"mat:{material}" for material in materials]

    def get_surface_area_by_cell_id(self, cell_id: int) -> List[float]:
        """Return the areas of surfaces bounding a cell."""
        self._require_unmodified_source("get_surface_area_by_cell_id")
        return core.get_surface_area_by_cell_id(
            self.filename,
            cell_id=cell_id,
            backend=self.backend,
        )

    def get_surface_area_by_material_name(self, material: str) -> List[float]:
        """Return areas of surfaces bounding volumes with a material."""
        self._require_unmodified_source("get_surface_area_by_material_name")
        return core.get_surface_area_by_material_name(
            self.filename,
            material=material,
            backend=self.backend,
        )

    def get_surface_area_by_surface_id(self) -> Dict[int, float]:
        """Return surface IDs mapped to their areas."""
        self._require_unmodified_source("get_surface_area_by_surface_id")
        return core.get_surface_area_by_surface_id(
            self.filename,
            backend=self.backend,
        )

    def get_surface_ids(self) -> List[int]:
        """Return the sorted surface IDs in the source file."""
        self._require_unmodified_source("get_surface_ids")
        return core.get_surface_ids(self.filename, backend=self.backend)

    def get_surface_ids_by_cell_id(self, cell_id: int) -> List[int]:
        """Return surface IDs bounding a cell."""
        self._require_unmodified_source("get_surface_ids_by_cell_id")
        return core.get_surface_ids_by_cell_id(
            self.filename,
            cell_id=cell_id,
            backend=self.backend,
        )

    def get_surface_ids_by_material_name(self, material: str) -> List[int]:
        """Return surface IDs bounding volumes with a material."""
        self._require_unmodified_source("get_surface_ids_by_material_name")
        return core.get_surface_ids_by_material_name(
            self.filename,
            material=material,
            backend=self.backend,
        )

    def get_surface_shared_status(self) -> Dict[int, Dict[str, list]]:
        """Return the cells and materials sharing each surface."""
        self._require_unmodified_source("get_surface_shared_status")
        return core.get_surface_shared_status(self.filename, backend=self.backend)

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

    def get_volumes_by_cell_id(self) -> Dict[int, float]:
        """Return cell IDs mapped to their geometric volumes."""
        self._require_unmodified_source("get_volumes_by_cell_id")
        return core.get_volumes_by_cell_id(self.filename, backend=self.backend)

    def get_volumes_by_cell_id_and_material_name(
        self,
    ) -> Dict[Tuple[int, str], float]:
        """Return cell and material pairs mapped to geometric volumes."""
        volume_sizes = self.get_volumes_by_cell_id()
        return {
            (volume_id, material): volume_sizes.get(volume_id, 0.0)
            for volume_id, material in self._data.volume_materials.items()
        }

    def get_volumes_by_material_name(self) -> Dict[str, float]:
        """Return material names mapped to total geometric volumes."""
        volume_sizes = self.get_volumes_by_cell_id()
        result: Dict[str, float] = {}
        for volume_id, material in self._data.volume_materials.items():
            result[material] = result.get(material, 0.0) + volume_sizes.get(
                volume_id, 0.0
            )
        return result

    def get_volumes_and_materials(self, remove_prefix: bool = True) -> Dict[int, str]:
        """Return volume IDs mapped to their material tags."""
        if remove_prefix:
            return dict(self._data.volume_materials)
        return {
            volume_id: f"mat:{material}"
            for volume_id, material in self._data.volume_materials.items()
        }

    def move(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Translate the in-memory geometry by the given offset."""
        offset = np.array([x, y, z])
        self._data.volume_data = {
            volume_id: (connectivity, coordinates + offset)
            for volume_id, (connectivity, coordinates) in self._data.volume_data.items()
        }
        self._modified = True

    def remove_materials(self, materials_to_remove: Union[str, List[str]]) -> List[str]:
        """Remove volumes with the specified materials from memory."""
        if isinstance(materials_to_remove, str):
            materials_to_remove = [materials_to_remove]

        available_materials = self.get_materials()
        matched = sorted(set(materials_to_remove) & set(available_materials))
        if not matched:
            raise ValueError(
                f"None of the specified materials {materials_to_remove} found in "
                f"{self.filename}. Available materials: {available_materials}"
            )

        volume_ids_to_remove = {
            volume_id
            for volume_id, material in self._data.volume_materials.items()
            if material in materials_to_remove
        }
        for volume_id in volume_ids_to_remove:
            self._data.volume_data.pop(volume_id, None)
            self._data.volume_materials.pop(volume_id, None)

        removed = set(matched)
        self._data.materials = [m for m in self._data.materials if m not in removed]

        self._modified = True
        return matched

    def remove_volumes(self, volume_ids_to_remove: Union[int, List[int]]) -> List[int]:
        """Remove the specified volume IDs from memory."""
        if isinstance(volume_ids_to_remove, int):
            volume_ids_to_remove = [volume_ids_to_remove]

        available_volume_ids = self.get_volumes()
        matched = sorted(set(volume_ids_to_remove) & set(available_volume_ids))
        if not matched:
            raise ValueError(
                f"None of the specified volume IDs {volume_ids_to_remove} found in "
                f"{self.filename}. Available volume IDs: {available_volume_ids}"
            )

        used_before = set(self._data.volume_materials.values())
        for volume_id in matched:
            self._data.volume_data.pop(volume_id, None)
            self._data.volume_materials.pop(volume_id, None)
        used_after = set(self._data.volume_materials.values())

        self._data.materials = [
            m for m in self._data.materials if m in used_after or m not in used_before
        ]

        self._modified = True
        return matched

    def rotate_around_axis(
        self,
        axis: Literal["x", "y", "z"] = "z",
        degrees: float = 90,
    ) -> None:
        """Rotate the in-memory geometry around a coordinate axis."""
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis {axis!r}. Must be one of 'x', 'y', or 'z'.")

        rotation = _rotation_matrix(axis, degrees)
        self._data.volume_data = {
            volume_id: (connectivity, coordinates @ rotation.T)
            for volume_id, (connectivity, coordinates) in self._data.volume_data.items()
        }
        self._modified = True

    def set_boundary_condition(
        self,
        surface_id: int,
        boundary_condition: str,
        output_filename: Optional[str] = None,
    ) -> str:
        """Set a boundary condition in the source file or a copied output file."""
        self._require_unmodified_source("set_boundary_condition")
        return core.set_boundary_condition(
            input_filename=self.filename,
            surface_id=surface_id,
            boundary_condition=boundary_condition,
            output_filename=output_filename,
            backend=self.backend,
        )

    def set_openmc_material_volumes(self, materials: object) -> None:
        """Set OpenMC material volumes using the source DAGMC geometry."""
        self._require_unmodified_source("set_openmc_material_volumes")
        core.set_openmc_material_volumes(
            materials,
            self.filename,
            backend=self.backend,
        )

    def write(self, output_filename: str) -> str:
        """Write the current in-memory geometry and materials to an h5m file."""
        orphans = sorted(set(self._data.volume_data) - set(self._data.volume_materials))
        if orphans:
            raise ValueError(
                f"Cannot write {output_filename}: volumes {orphans} have no "
                "material group. Assign a material or remove these volumes."
            )
        _write_h5m(
            output_filename,
            self._data.volume_data,
            self._data.volume_materials,
        )
        return output_filename
