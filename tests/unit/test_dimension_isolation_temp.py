from pathlib import Path
from types import SimpleNamespace

from tests.helpers.dimension_isolation import _isolated_command


def test_dimension_children_do_not_share_pytest_temporary_data():
    item = SimpleNamespace(nodeid="test_example.py::test_example", config=SimpleNamespace())
    for directory in ("/unique-child-one", "/unique-child-two"):
        report = Path(directory) / "report.xml"
        command = _isolated_command(item, report)
        assert f"--basetemp={report.parent / 'pytest'}" in command
