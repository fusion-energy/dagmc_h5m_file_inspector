import json
import os
import subprocess
import unittest
import urllib.request
from pathlib import Path
import tarfile


class TestReactor(unittest.TestCase):
    def setUp(self):

        if not Path("tests/v0.0.1.tar.gz").is_file():
            url = "https://github.com/Shimwell/fusion_example_for_openmc_using_paramak/archive/refs/tags/v0.0.1.tar.gz"
            urllib.request.urlretrieve(url, "tests/v0.0.1.tar.gz")

        tar = tarfile.open("tests/v0.0.1.tar.gz", "r:gz")
        tar.extractall("tests")
        tar.close()

    def test_volume_finding(self):
        """Tests command runs and produces an output files with the correct contents"""
        os.system("rm vols.json")
        os.system(
            "inspect-dagmc-h5m-file -i tests/fusion_example_for_openmc_using_paramak-0.0.1/dagmc.h5m -v -o vols.json"
        )
        assert Path("vols.json").is_file()
        with open("vols.json") as jsonFile:
            jsonObject = json.load(jsonFile)
        assert jsonObject == {
            "volumes": [
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
        }

    def test_material_finding(self):
        """Tests command runs and produces an output files with the correct contents"""
        os.system("rm vols.json")
        os.system(
            "inspect-dagmc-h5m-file -i tests/fusion_example_for_openmc_using_paramak-0.0.1/dagmc.h5m -m -o mats.json"
        )
        assert Path("mats.json").is_file()
        with open("mats.json") as jsonFile:
            jsonObject = json.load(jsonFile)
        assert jsonObject == {
            "materials": ["Vacuum", "copper", "flibe", "graveyard", "steel", "tungsten"]
        }

    def test_both_finding(self):
        """Tests command runs and produces an output files with the correct contents"""
        os.system("rm vols.json")
        os.system(
            "inspect-dagmc-h5m-file -i tests/fusion_example_for_openmc_using_paramak-0.0.1/dagmc.h5m -b -o both.json"
        )
        assert Path("both.json").is_file()
        with open("both.json") as jsonFile:
            jsonObject = json.load(jsonFile)
        assert jsonObject == {
            "both": {
                "1": "tungsten",
                "2": "steel",
                "3": "steel",
                "4": "flibe",
                "5": "flibe",
                "6": "tungsten",
                "7": "tungsten",
                "8": "steel",
                "9": "steel",
                "10": "steel",
                "11": "copper",
                "12": "copper",
                "13": "copper",
                "14": "copper",
                "15": "copper",
                "16": "copper",
                "17": "steel",
                "18": "steel",
                "19": "steel",
                "20": "steel",
                "21": "graveyard",
                "22": "Vacuum",
            }
        }

    def test_no_user_output(self):
        """Tests command runs and produces an output files with the correct contents"""
        os.system("rm empty.json")
        os.system(
            "inspect-dagmc-h5m-file -i tests/fusion_example_for_openmc_using_paramak-0.0.1/dagmc.h5m -o empty.json"
        )
        assert Path("empty.json").is_file() == False

    # todo improve tests to check commands run correctly

    def test_help(self):
        os.system("inspect-dagmc-h5m-file --help")

    def test_no_user_input(self):
        os.system("inspect-dagmc-h5m-file")


if __name__ == "__main__":
    unittest.main()
