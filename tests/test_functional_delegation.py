from unittest.mock import patch

import pytest

import dagmc_h5m_file_inspector as di
from dagmc_h5m_file_inspector import dagmc_file


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
@pytest.mark.parametrize(
    "operation",
    [
        lambda input_file, output_file, backend: di.remove_materials(
            input_file, output_file, "box_a", backend=backend
        ),
        lambda input_file, output_file, backend: di.remove_volumes(
            input_file, output_file, 1, backend=backend
        ),
        lambda input_file, output_file, backend: di.move(
            input_file, x=10.0, output=output_file, backend=backend
        ),
        lambda input_file, output_file, backend: di.rotate_around_axis(
            input_file,
            axis="z",
            degrees=90,
            output=output_file,
            backend=backend,
        ),
    ],
    ids=["remove-materials", "remove-volumes", "move", "rotate"],
)
def test_functional_mutations_load_once(separated_boxes, tmp_path, backend, operation):
    output = str(tmp_path / f"output_{backend}.h5m")

    with patch.object(
        dagmc_file, "_load_dagmc_data", wraps=dagmc_file._load_dagmc_data
    ) as load_mock:
        operation(separated_boxes["filename"], output, backend)

    assert load_mock.call_count == 1


@pytest.mark.parametrize("backend", ["h5py", "pymoab"])
def test_combine_loads_each_input_once(
    separated_boxes, cube_geometry, tmp_path, backend
):
    output = str(tmp_path / f"combined_{backend}.h5m")

    with patch(
        "dagmc_h5m_file_inspector.core._load_dagmc_data",
        wraps=dagmc_file._load_dagmc_data,
    ) as load_mock:
        di.combine_h5m_files(
            [separated_boxes["filename"], cube_geometry["filename"]],
            output,
            backend=backend,
        )

    assert load_mock.call_count == 2
    assert di.get_volumes(output) == [1, 2, 3]
