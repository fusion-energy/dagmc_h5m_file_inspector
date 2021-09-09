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
            "tests/fusion_example_for_openmc_using_paramak-0.0.1/dagmc.h5m"
        )

        assert dict_of_vol_and_mats == {
            1: "mat:tungsten",
            6: "mat:tungsten",
            7: "mat:tungsten",
            2: "mat:steel",
            3: "mat:steel",
            8: "mat:steel",
            9: "mat:steel",
            10: "mat:steel",
            17: "mat:steel",
            18: "mat:steel",
            19: "mat:steel",
            20: "mat:steel",
            4: "mat:flibe",
            5: "mat:flibe",
            11: "mat:copper",
            12: "mat:copper",
            13: "mat:copper",
            14: "mat:copper",
            15: "mat:copper",
            16: "mat:copper",
            21: "mat:graveyard",
            22: "mat:Vacuum",
        }

    def test_volume_extraction(self):
        """Extracts the volume ids from a dagmc file and checks the contents
        match the expected contents"""

        dict_of_vol_and_mats = di.get_volumes_from_h5m(
            "tests/fusion_example_for_openmc_using_paramak-0.0.1/dagmc.h5m"
        )

        assert dict_of_vol_and_mats == [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
        ]

    def test_material_extraction(self):
        """Extracts the materials tags from a dagmc file and checks the
        contents match the expected contents"""

        dict_of_vol_and_mats = di.get_materials_from_h5m(
            "tests/fusion_example_for_openmc_using_paramak-0.0.1/dagmc.h5m"
        )

        assert dict_of_vol_and_mats == [
            "mat:Vacuum",
            "mat:copper",
            "mat:flibe",
            "mat:graveyard",
            "mat:steel",
            "mat:tungsten",
        ]
