import tarfile
import unittest
import urllib.request
from pathlib import Path
import dagmc_h5m_file_inspector as di


class TestApiUsage(unittest.TestCase):
    def setUp(self):

        if not Path("tests/v0.0.1.tar.gz").is_file():
            url = "https://github.com/Shimwell/fusion_example_for_openmc_using_paramak/archive/refs/tags/v0.0.1.tar.gz"
            urllib.request.urlretrieve(url, "tests/v0.0.1.tar.gz")

        tar = tarfile.open("tests/v0.0.1.tar.gz", "r:gz")
        tar.extractall("tests")
        tar.close()

    def test_volume_and_material_extraction(self):
        """Extracts the volume numbers and material ids from a dagmc file and
        checks the contents match the expected contents"""

        dict_of_vol_and_mats = di.get_volumes_and_materials_from_h5m(
            "tests/dagmc.h5m")

        print(dict_of_vol_and_mats)

        assert dict_of_vol_and_mats == {
            1: "mat:pf_coil_mat",
            2: "mat:pf_coil_mat",
            3: "mat:pf_coil_mat",
            4: "mat:pf_coil_mat",
            5: "mat:pf_coil_case_mat",
            6: "mat:pf_coil_case_mat",
            7: "mat:pf_coil_case_mat",
            8: "mat:pf_coil_case_mat",
            9: "mat:inboard_tf_coils_mat",
            10: "mat:center_column_shield_mat",
            11: "mat:firstwall_mat",
            12: "mat:blanket_mat",
            13: "mat:blanket_rear_wall_mat",
            14: "mat:divertor_mat",
            15: "mat:divertor_mat",
            16: "mat:tf_coil_mat",
            17: "mat:graveyard",
        }
