import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transcribe import main as _root_main


def main(args_list=None):
    return _root_main(args_list)


if __name__ == "__main__":
    sys.exit(main())
