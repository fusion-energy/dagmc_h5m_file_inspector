import cadquery as cq
import numpy as np
import pytest
from cad_to_dagmc import CadToDagmc


def create_touching_boxes(tmp_path):
    """Create two boxes that touch at a shared face.

    small_box: 10x10x10 cube centered at origin
    big_box: 20x20x20 cube centered at x=15 (touches small_box at x=5)
    """
    width1 = 10  # small_box dimensions
    width2 = 20  # big_box dimensions

    assembly = cq.Assembly()
    cuboid1 = cq.Workplane().box(width1, width1, width1)
    assembly.add(cuboid1, name="small_box")
    cuboid2 = (
        cq.Workplane().moveTo(0.5 * width1 + 0.5 * width2).box(width2, width2, width2)
    )
    assembly.add(cuboid2, name="big_box")

    filename = str(tmp_path / "dagmc.h5m")
    my_model = CadToDagmc()
    my_model.add_cadquery_object(
        cadquery_object=assembly, material_tags="assembly_names"
    )
    my_model.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=filename
    )

    return {
        "filename": filename,
        "volumes": [1, 2],
        "materials": ["big_box", "small_box"],
        "materials_with_prefix": ["mat:big_box", "mat:small_box"],
        "volumes_and_materials": {1: "small_box", 2: "big_box"},
        "volumes_and_materials_with_prefix": {1: "mat:small_box", 2: "mat:big_box"},
        "expected_volume_sizes": {1: width1**3, 2: width2**3},
        "lower_left": np.array([-5.0, -10.0, -10.0]),
        "upper_right": np.array([25.0, 10.0, 10.0]),
        "small_box_lower_left": np.array([-5.0, -5.0, -5.0]),
        "small_box_upper_right": np.array([5.0, 5.0, 5.0]),
        "big_box_lower_left": np.array([5.0, -10.0, -10.0]),
        "big_box_upper_right": np.array([25.0, 10.0, 10.0]),
    }


def create_separated_boxes(tmp_path):
    """Create two boxes that do not touch (separated by a gap).

    box_a: 10x10x10 cube centered at origin
    box_b: 10x10x10 cube centered at x=20 (gap of 5 units between them)
    """
    width = 10  # both boxes have same dimensions
    gap = 5  # gap between boxes

    assembly = cq.Assembly()
    cuboid1 = cq.Workplane().box(width, width, width)
    assembly.add(cuboid1, name="box_a")
    # Position box_b so there's a gap: box_a goes from -5 to 5, box_b from 10 to 20
    cuboid2 = cq.Workplane().moveTo(width + gap).box(width, width, width)
    assembly.add(cuboid2, name="box_b")

    filename = str(tmp_path / "dagmc_separated.h5m")
    my_model = CadToDagmc()
    my_model.add_cadquery_object(
        cadquery_object=assembly, material_tags="assembly_names"
    )
    my_model.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=filename
    )

    return {
        "filename": filename,
        "volumes": [1, 2],
        "materials": ["box_a", "box_b"],
        "materials_with_prefix": ["mat:box_a", "mat:box_b"],
        "volumes_and_materials": {1: "box_a", 2: "box_b"},
        "volumes_and_materials_with_prefix": {1: "mat:box_a", 2: "mat:box_b"},
        "expected_volume_sizes": {1: width**3, 2: width**3},
        "lower_left": np.array([-5.0, -5.0, -5.0]),
        "upper_right": np.array([20.0, 5.0, 5.0]),
        "box_a_lower_left": np.array([-5.0, -5.0, -5.0]),
        "box_a_upper_right": np.array([5.0, 5.0, 5.0]),
        "box_b_lower_left": np.array([10.0, -5.0, -5.0]),
        "box_b_upper_right": np.array([20.0, 5.0, 5.0]),
    }


@pytest.fixture(scope="session")
def touching_boxes(tmp_path_factory):
    """Fixture providing the touching boxes geometry."""
    tmp_path = tmp_path_factory.mktemp("touching")
    return create_touching_boxes(tmp_path)


@pytest.fixture(scope="session")
def separated_boxes(tmp_path_factory):
    """Fixture providing the separated boxes geometry."""
    tmp_path = tmp_path_factory.mktemp("separated")
    return create_separated_boxes(tmp_path)


def create_cube_geometry(tmp_path):
    """Create a 10x10x10 cube with material 'cube'."""
    cube = cq.Workplane().box(10, 10, 10)

    filename = str(tmp_path / "cube.h5m")
    my_model = CadToDagmc()
    my_model.add_cadquery_object(cadquery_object=cube, material_tags=["cube"])
    my_model.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=filename
    )

    return {
        "filename": filename,
        "cell_id": 1,
        "material": "cube",
        "expected_num_surfaces": 6,
        "expected_surface_area_each": 100.0,
        "expected_total_surface_area": 600.0,
    }


def create_sphere_geometry(tmp_path):
    """Create a sphere with radius 5 and material 'sphere'."""
    sphere = cq.Workplane().sphere(5)

    filename = str(tmp_path / "sphere.h5m")
    my_model = CadToDagmc()
    my_model.add_cadquery_object(cadquery_object=sphere, material_tags=["sphere"])
    my_model.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=filename
    )

    return {
        "filename": filename,
        "cell_id": 1,
        "material": "sphere",
        "expected_total_surface_area": 4 * np.pi * 25,  # 4*pi*r^2
    }


def create_rectangle_geometry(tmp_path):
    """Create a 10x20x30 cuboid with material 'rectangle'."""
    cuboid = cq.Workplane().box(10, 20, 30)

    filename = str(tmp_path / "rectangle.h5m")
    my_model = CadToDagmc()
    my_model.add_cadquery_object(cadquery_object=cuboid, material_tags=["rectangle"])
    my_model.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=filename
    )

    return {
        "filename": filename,
        "cell_id": 1,
        "material": "rectangle",
        "expected_num_surfaces": 6,
        "expected_sorted_areas": [200.0, 200.0, 300.0, 300.0, 600.0, 600.0],
        "expected_total_surface_area": 2200.0,
    }


def create_cylinder_geometry(tmp_path):
    """Create a cylinder with height 20, radius 5 and material 'cylinder'.

    3 surfaces: top cap, bottom cap (flat), and lateral (curved).
    - Each cap area: pi * r^2 = 25*pi ≈ 78.54
    - Lateral area: 2*pi*r * h = 200*pi ≈ 628.32
    - Total: 2*pi*r*(r + h) = 250*pi ≈ 785.40
    """
    cyl = cq.Workplane().cylinder(20, 5)

    filename = str(tmp_path / "cylinder.h5m")
    my_model = CadToDagmc()
    my_model.add_cadquery_object(cadquery_object=cyl, material_tags=["cylinder"])
    my_model.export_dagmc_h5m_file(
        min_mesh_size=0.5, max_mesh_size=1.0e6, filename=filename
    )

    r = 5
    h = 20
    return {
        "filename": filename,
        "cell_id": 1,
        "material": "cylinder",
        "expected_num_surfaces": 3,
        "expected_cap_area": np.pi * r**2,  # ≈ 78.54
        "expected_lateral_area": 2 * np.pi * r * h,  # ≈ 628.32
        "expected_total_surface_area": 2 * np.pi * r * (r + h),  # ≈ 785.40
    }


@pytest.fixture(scope="session")
def cylinder_geometry(tmp_path_factory):
    """Fixture providing the cylinder geometry."""
    tmp_path = tmp_path_factory.mktemp("cylinder")
    return create_cylinder_geometry(tmp_path)


@pytest.fixture(scope="session")
def cube_geometry(tmp_path_factory):
    """Fixture providing the cube geometry."""
    tmp_path = tmp_path_factory.mktemp("cube")
    return create_cube_geometry(tmp_path)


@pytest.fixture(scope="session")
def sphere_geometry(tmp_path_factory):
    """Fixture providing the sphere geometry."""
    tmp_path = tmp_path_factory.mktemp("sphere")
    return create_sphere_geometry(tmp_path)


@pytest.fixture(scope="session")
def rectangle_geometry(tmp_path_factory):
    """Fixture providing the rectangle geometry."""
    tmp_path = tmp_path_factory.mktemp("rectangle")
    return create_rectangle_geometry(tmp_path)


def create_grouped_boxes(tmp_path):
    """Create two touching boxes and add non-material groups.

    Builds on the touching boxes geometry (which already has ``mat:`` groups)
    and uses pymoab to add extra non-material groups so the file contains group
    membership beyond the material tags:

    - ``component:small_box`` -> cell 1
    - ``component:big_box`` -> cell 2
    - ``assembly:all`` -> cells 1 and 2

    Cell 1 is the small_box (mat:small_box) and cell 2 is the big_box
    (mat:big_box), matching the touching boxes geometry.
    """
    from pymoab import core, types

    base = create_touching_boxes(tmp_path)
    filename = base["filename"]

    mbcore = core.Core()
    mbcore.load_file(filename)

    category_tag = mbcore.tag_get_handle(types.CATEGORY_TAG_NAME)
    name_tag = mbcore.tag_get_handle(types.NAME_TAG_NAME)
    id_tag = mbcore.tag_get_handle(types.GLOBAL_ID_TAG_NAME)

    volume_ents = mbcore.get_entities_by_type_and_tag(
        0, types.MBENTITYSET, category_tag, ["Volume"]
    )
    volume_by_cell_id = {}
    for vol in volume_ents:
        cell_id = mbcore.tag_get_data(id_tag, vol)[0][0].item()
        volume_by_cell_id[cell_id] = vol

    def add_group(name, cell_ids):
        group_set = mbcore.create_meshset()
        mbcore.tag_set_data(category_tag, group_set, "Group")
        mbcore.tag_set_data(name_tag, group_set, name)
        for cell_id in cell_ids:
            mbcore.add_entity(group_set, volume_by_cell_id[cell_id])

    add_group("component:small_box", [1])
    add_group("component:big_box", [2])
    add_group("assembly:all", [1, 2])

    mbcore.write_file(filename)

    base["cell_ids_by_group_name"] = {
        "component:small_box": [1],
        "component:big_box": [2],
        "assembly:all": [1, 2],
    }
    base["groups_by_cell_id"] = {
        1: ["assembly:all", "component:small_box"],
        2: ["assembly:all", "component:big_box"],
    }
    return base


@pytest.fixture(scope="session")
def grouped_boxes(tmp_path_factory):
    """Fixture providing two boxes with extra non-material groups."""
    tmp_path = tmp_path_factory.mktemp("grouped")
    return create_grouped_boxes(tmp_path)
