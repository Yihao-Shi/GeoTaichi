import subprocess
import tempfile
from pathlib import Path

HOOK = Path(__file__).with_name("pre-commit")


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def test_pre_commit():
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        git(repo, "init", "-q")
        source = repo / "example.py"
        source.write_text("values=[1,2,3]\n")
        git(repo, "add", "example.py")

        subprocess.run([HOOK], cwd=repo, check=True)
        assert git(repo, "show", ":example.py") == "values = [1, 2, 3]\n"

        source.write_text("values=[4,5,6]\n")
        git(repo, "add", "example.py")
        source.write_text("values=[7,8,9]\n")
        result = subprocess.run([HOOK], cwd=repo, capture_output=True, text=True)

        assert result.returncode == 1
        assert git(repo, "show", ":example.py") == "values=[4,5,6]\n"
        assert source.read_text() == "values=[7,8,9]\n"


if __name__ == "__main__":
    test_pre_commit()
