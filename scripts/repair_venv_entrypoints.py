from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("venv", type=Path)
    args = parser.parse_args()
    venv = args.venv.resolve()
    bin_dir = venv / "bin"
    desired = f"#!{venv}/bin/python\n"
    repaired: list[Path] = []

    for path in bin_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        first, separator, rest = raw.partition(b"\n")
        if not separator:
            continue
        line = first.decode("utf-8", errors="ignore")
        if line.startswith("#!/opt/venv/bin/python") or line.startswith(
            "!/opt/venv/bin/python.*#!/opt/venv/bin/python"
        ):
            path.write_bytes(desired.encode() + rest)
            repaired.append(path)

    print({"venv": str(venv), "repaired_count": len(repaired)})
    for path in repaired:
        print(path)


if __name__ == "__main__":
    main()
