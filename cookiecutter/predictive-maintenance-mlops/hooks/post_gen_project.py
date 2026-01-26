import sys
import shutil
from pathlib import Path

def rm_tree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)

def rm_file(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass

def main() -> None:
    # Avoid creating bytecode during hook execution
    sys.dont_write_bytecode = True

    # Remove any caches that may have been created anyway
    for d in Path(".").rglob("__pycache__"):
        rm_tree(d)

    for f in Path(".").rglob("*.pyc"):
        rm_file(f)

    print("OK: project generated (clean)")

if __name__ == "__main__":
    main()
