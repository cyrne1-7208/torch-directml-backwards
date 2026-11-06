"""Stage the CPU-checked wheels and GPU checks for one CI revision."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build import targets
from ci.changes import CHECKS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKS = tuple(sorted(CHECKS))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(artifacts, output, revision, abis=None, checks=DEFAULT_CHECKS):
    if not checks or not set(checks) <= CHECKS:
        raise ValueError("Select at least one known check group")
    expected = {
        f"{t['abi']}-torch{t['torch']}-win_amd64"
        for t in targets()
        if abis is None or t["abi"] in abis
    }
    manifests = [
        json.loads(p.read_text(encoding="utf-8-sig")) | {"directory": p.parent}
        for p in artifacts.glob("*/manifest.json")
    ]
    if abis is not None:
        manifests = [m for m in manifests if m["abi"] in abis]
    if len(manifests) != len(expected) or {m["id"] for m in manifests} != expected:
        raise ValueError(
            "CPU artifacts must contain every selected target exactly once"
        )
    output.mkdir(parents=True, exist_ok=False)
    result = []
    constraints = None
    for manifest in sorted(manifests, key=lambda m: m["id"]):
        directory = manifest.pop("directory")
        if manifest["revision"] != revision or manifest.get("cpu_checked") is not True:
            raise ValueError(
                f"Missing CPU validation for this revision: {manifest['id']}"
            )
        wheel = directory / manifest["wheel"]
        if wheel.parent != directory or digest(wheel) != manifest["wheel_sha256"]:
            raise ValueError(f"Wheel differs from its CPU artifact: {manifest['id']}")
        contents = (directory / "constraints.txt").read_bytes()
        if constraints is not None and contents != constraints:
            raise ValueError("CPU artifacts have different dependency constraints")
        constraints = contents
        destination = output / "wheels" / wheel.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(wheel, destination)
        manifest["wheel"] = destination.relative_to(output).as_posix()
        native = output / "native" / manifest["id"]
        native.mkdir(parents=True)
        for name in ("native_checks.dll", "native_checks.json"):
            shutil.copyfile(directory / name, native / name)
        metadata = json.loads(
            (native / "native_checks.json").read_text(encoding="utf-8")
        )
        if metadata["native_sha256"] != manifest["native_sha256"]:
            raise ValueError(
                f"Native checks belong to a different build: {manifest['id']}"
            )
        manifest["native_checks"] = (
            (native / "native_checks.dll").relative_to(output).as_posix()
        )
        result.append(manifest)
    (output / "constraints.txt").write_bytes(constraints)
    for source in (
        ROOT / "tests/common.py",
        *(ROOT / f"tests/{name}.py" for name in sorted(CHECKS)),
        ROOT / "tools/ci/windows.ps1",
        ROOT / "tools/ci/LICENSE",
        ROOT / "tools/ci/LICENSE.runner-images",
    ):
        destination = (
            output / ("tests" if source.parent.name == "tests" else "") / source.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    hashes = {
        p.relative_to(output).as_posix(): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "targets": result,
                "checks": checks,
                "hashes": hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".work/ci/artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / ".work/ci/payload")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--abis", nargs="*", choices=sorted({t["abi"] for t in targets()})
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=sorted(CHECKS),
        default=sorted(CHECKS),
    )
    args = parser.parse_args()
    try:
        prepare(
            args.artifacts.resolve(),
            args.output.resolve(),
            args.revision,
            args.abis or None,
            args.checks,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
