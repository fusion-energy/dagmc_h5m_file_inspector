import pytest
import numpy as np
import cadquery as cq
from cad_to_dagmc import CadToDagmc


def create_touching_boxes():
    """Create two boxes that touch at a shared face.

    small_box: 10x10x10 cube centered at origin
    big_box: 20x20x20 cube centered at x=15 (touches small_box at x=5)
    """
    width1 = 10  # small_box dimensions
    width2 = 20  # big_box dimensions

    assembly = cq.Assembly()
    cuboid1 = cq.Workplane().box(width1, width1, width1)
    assembly.add(cuboid1, name='small_box')
    cuboid2 = cq.Workplane().moveTo(0.5 * width1 + 0.5 * width2).box(width2, width2, width2)
    assembly.add(cuboid2, name='big_box')

    my_model = CadToDagmc()
    my_model.add_cadquery_object(cadquery_object=assembly, material_tags="assembly_names")
    my_model.export_dagmc_h5m_file(min_mesh_size=0.5, max_mesh_size=1.0e6, filename='tests/dagmc.h5m')

    return {
        'filename': 'tests/dagmc.h5m',
        'volumes': [1, 2],
        'materials': ['big_box', 'small_box'],
        'materials_with_prefix': ['mat:big_box', 'mat:small_box'],
        'volumes_and_materials': {1: 'small_box', 2: 'big_box'},
        'volumes_and_materials_with_prefix': {1: 'mat:small_box', 2: 'mat:big_box'},
        'expected_volume_sizes': {1: width1 ** 3, 2: width2 ** 3},
        'lower_left': np.array([-5.0, -10.0, -10.0]),
        'upper_right': np.array([25.0, 10.0, 10.0]),
    }


def create_separated_boxes():
    """Create two boxes that do not touch (separated by a gap).

    box_a: 10x10x10 cube centered at origin
    box_b: 10x10x10 cube centered at x=20 (gap of 5 units between them)
    """
    width = 10  # both boxes have same dimensions
    gap = 5  # gap between boxes

    assembly = cq.Assembly()
    cuboid1 = cq.Workplane().box(width, width, width)
    assembly.add(cuboid1, name='box_a')
    # Position box_b so there's a gap: box_a goes from -5 to 5, box_b from 10 to 20
    cuboid2 = cq.Workplane().moveTo(width + gap).box(width, width, width)
    assembly.add(cuboid2, name='box_b')

    my_model = CadToDagmc()
    my_model.add_cadquery_object(cadquery_object=assembly, material_tags="assembly_names")
    my_model.export_dagmc_h5m_file(min_mesh_size=0.5, max_mesh_size=1.0e6, filename='tests/dagmc_separated.h5m')

    return {
        'filename': 'tests/dagmc_separated.h5m',
        'volumes': [1, 2],
        'materials': ['box_a', 'box_b'],
        'materials_with_prefix': ['mat:box_a', 'mat:box_b'],
        'volumes_and_materials': {1: 'box_a', 2: 'box_b'},
        'volumes_and_materials_with_prefix': {1: 'mat:box_a', 2: 'mat:box_b'},
        'expected_volume_sizes': {1: width ** 3, 2: width ** 3},
        'lower_left': np.array([-5.0, -5.0, -5.0]),
        'upper_right': np.array([20.0, 5.0, 5.0]),
    }


@pytest.fixture(scope="session")
def touching_boxes():
    """Fixture providing the touching boxes geometry."""
    return create_touching_boxes()


@pytest.fixture(scope="session")
def separated_boxes():
    """Fixture providing the separated boxes geometry."""
    return create_separated_boxes()
