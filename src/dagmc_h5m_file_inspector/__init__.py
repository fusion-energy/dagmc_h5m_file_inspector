from importlib.metadata import version

__version__ = version("dagmc_h5m_file_inspector")

from .core import (
    BoundingBox,
    combine_h5m_files,
    convert_h5m_to_vtkhdf,
    get_bounding_box_from_h5m,
    get_materials_from_h5m,
    get_surface_area_by_cell_id,
    get_surface_area_by_material_name,
    get_surface_shared_status,
    get_triangle_conn_and_coords_by_volume,
    get_volumes_and_materials_from_h5m,
    get_volumes_from_h5m,
    get_volumes_from_h5m_by_cell_id,
    get_volumes_from_h5m_by_cell_id_and_material_name,
    get_volumes_from_h5m_by_material_name,
    move,
    remove_materials_from_h5m,
    rotate_around_axis,
    set_openmc_material_volumes_from_h5m,
)

__all__ = [
    "BoundingBox",
    "combine_h5m_files",
    "convert_h5m_to_vtkhdf",
    "get_bounding_box_from_h5m",
    "get_materials_from_h5m",
    "get_surface_area_by_cell_id",
    "get_surface_area_by_material_name",
    "get_surface_shared_status",
    "get_triangle_conn_and_coords_by_volume",
    "get_volumes_and_materials_from_h5m",
    "get_volumes_from_h5m",
    "get_volumes_from_h5m_by_cell_id",
    "get_volumes_from_h5m_by_cell_id_and_material_name",
    "get_volumes_from_h5m_by_material_name",
    "move",
    "remove_materials_from_h5m",
    "rotate_around_axis",
    "set_openmc_material_volumes_from_h5m",
]
