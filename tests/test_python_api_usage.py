import cadquery as cq
import dagmc_h5m_file_inspector as di
from cad_to_dagmc import CadToDagmc


width1 = 10
width2 = 20
assembly = cq.Assembly()
cuboid1 = cq.Workplane().box(width1, width1, width1)
assembly.add(cuboid1, name='small_box')
cuboid2 = cq.Workplane().moveTo(0.5*width1+ 0.5*width2).box(width2, width2, width2)
assembly.add(cuboid2, name='big_box')

my_model = CadToDagmc()
my_model.add_cadquery_object(cadquery_object=assembly, material_tags="assembly_names")
my_model.export_dagmc_h5m_file(min_mesh_size=0.5, max_mesh_size=1.0e6, filename='tests/dagmc.h5m')


def test_volume_and_material_extraction_without_stripped_prefix():
    """Extracts the volume numbers and material ids from a dagmc file and
    checks the contents match the expected contents"""

    dict_of_vol_and_mats = di.get_volumes_and_materials_from_h5m(
        filename="tests/dagmc.h5m",
        remove_prefix=False,
    )

    assert dict_of_vol_and_mats == {
        1: "mat:small_box",
        2: "mat:big_box",
    }

def test_volume_and_material_extraction_remove_prefix():
    """Extracts the volume numbers and material ids from a dagmc file and
    checks the contents match the expected contents"""

    dict_of_vol_and_mats = di.get_volumes_and_materials_from_h5m(
        "tests/dagmc.h5m"
    )

    assert dict_of_vol_and_mats == {
        1: "small_box",
        2: "big_box",
    }

def test_volume_extraction():
    """Extracts the volume ids from a dagmc file and checks the contents
    match the expected contents"""

    dict_of_vol_and_mats = di.get_volumes_from_h5m(
        "tests/dagmc.h5m"
    )

    assert dict_of_vol_and_mats == [
        1,
        2,
    ]

def test_material_extraction_no_remove_prefix():
    """Extracts the materials tags from a dagmc file and checks the
    contents match the expected contents"""

    dict_of_vol_and_mats = di.get_materials_from_h5m(
        filename="tests/dagmc.h5m",
        remove_prefix=False,
    )

    assert dict_of_vol_and_mats == [
        "mat:big_box",
        "mat:small_box",
    ]

def test_material_extraction_remove_prefix():
    """Extracts the materials tags from a dagmc file and checks the
    contents match the expected contents"""

    dict_of_vol_and_mats = di.get_materials_from_h5m(
        "tests/dagmc.h5m"
    )

    assert dict_of_vol_and_mats == [
        "big_box",
        "small_box",
    ]

def test_fail_with_missing_input_files():
    """Calls functions without necessary input files to check if error
    handeling is working"""

    def test_missing_file_error_handling():
        """Attempts to get info when the h5m file does not exist which
        should fail with a FileNotFoundError"""

        di.get_volumes_and_materials_from_h5m(
            filename="non_existant.h5m",
        ).assertRaises(FileNotFoundError, test_missing_file_error_handling)
