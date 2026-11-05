"""Build a target wheel with the reviewed upstream inputs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

from native_abi import PROFILES, load_profile

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, expected_hash):
    if destination.exists():
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_hash:
            raise ValueError(f"Unexpected SHA256 for {destination}: {digest}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_hash:
            raise ValueError(f"Unexpected SHA256 for {url}: {digest}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def torch_url(version, abi, expected_hash):
    filename = f"torch-{version}-{abi}-{abi}-win_amd64.whl"
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/torch/{version}/json", timeout=30
    ) as response:
        files = json.load(response)["urls"]
    entry = next(item for item in files if item["filename"] == filename)
    if entry["digests"]["sha256"] != expected_hash:
        raise ValueError(f"PyPI wheel differs from the reviewed profile: {filename}")
    return entry["url"]


def targets():
    result = []
    for path in sorted(PROFILES.glob("cp*.toml")):
        profile = load_profile(path.stem)
        for version in profile["target"]["wheels"]:
            result.append(
                {
                    "abi": path.stem,
                    "torch": version,
                    "python": profile["python"]["sdk"]["version"],
                }
            )
    return result


def build(args):
    profile = load_profile(args.abi)
    torch_hash = profile["target"]["wheels"][args.torch]
    target_id = f"{args.abi}-torch{args.torch}-win_amd64"
    work = (args.work_dir or ROOT / ".work" / target_id).resolve()
    output = (args.output_dir or ROOT / "dist" / target_id).resolve()
    build_dir = work / "build"
    if build_dir.exists() or output.exists():
        raise FileExistsError(
            "Choose new build and output directories; existing artifacts are not overwritten"
        )

    source = profile["source"]
    source_path = (
        ROOT
        / "reference/torch-directml"
        / Path(urllib.parse.urlsplit(source["url"]).path).name
    )
    download(source["url"], source_path, source["sha256"])
    url = torch_url(args.torch, args.abi, torch_hash)
    torch_path = ROOT / "reference/pytorch" / Path(urllib.parse.urlsplit(url).path).name
    download(url, torch_path, torch_hash)
    sdk = profile["python"]["sdk"]
    sdk_name = f"python.{sdk['version']}.nupkg"
    sdk_url = (
        f"https://api.nuget.org/v3-flatcontainer/python/{sdk['version']}/{sdk_name}"
    )
    sdk_path = ROOT / "reference/cpython" / sdk["version"] / sdk_name
    download(sdk_url, sdk_path, sdk["sha256"])

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = config["tool"]["torch-directml"]
    runtime_name = f"microsoft.ai.directml.{runtime['directml-version']}.nupkg"
    runtime_url = f"https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/{runtime['directml-version']}/{runtime_name}"
    runtime_path = ROOT / "reference/directml" / runtime_name
    download(runtime_url, runtime_path, runtime["directml-sha256"])

    command = [
        sys.executable,
        str(ROOT / "tools/build_native.py"),
        str(source_path),
        str(torch_path),
        "--python-sdk",
        str(sdk_path),
        "--output-dir",
        str(build_dir),
    ]
    if args.windows_sdk:
        command += ["--windows-sdk", str(args.windows_sdk)]
    subprocess.run(command, check=True)
    native = build_dir / f"torch_directml_native.{args.abi}-win_amd64.pyd"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/native_abi.py"),
            "patch",
            str(source_path),
            str(torch_path),
            "--python-sdk",
            str(sdk_path),
            "--native-code",
            str(build_dir / "native-code.bin"),
            "--output",
            str(native),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build_wheel.py"),
            "--native",
            str(native),
            "--directml-package",
            str(runtime_path),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    (wheel,) = output.glob("*.whl")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with native.open("rb") as stream:
        native_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "revision": args.revision,
        "id": target_id,
        "abi": args.abi,
        "python": sdk["version"],
        "python_url": sdk_url,
        "python_sha256": sdk["sha256"],
        "torch": args.torch,
        "torch_url": f"{url}#sha256={torch_hash}",
        "wheel": wheel.name,
        "wheel_sha256": digest,
        "native_sha256": native_hash,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "constraints.txt").open("w", encoding="utf-8") as stream:
        subprocess.run(
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--no-emit-package",
                "torch",
                "--no-hashes",
            ],
            cwd=ROOT,
            stdout=stream,
            check=True,
        )
    print(wheel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("matrix", help="Print the reviewed GitHub Actions build matrix")
    make = commands.add_parser(
        "build", help="Download reviewed inputs and build one wheel"
    )
    make.add_argument(
        "abi", choices=[path.stem for path in sorted(PROFILES.glob("cp*.toml"))]
    )
    make.add_argument("torch")
    make.add_argument("--windows-sdk", type=Path)
    make.add_argument("--work-dir", type=Path)
    make.add_argument("--output-dir", type=Path)
    make.add_argument("--revision", default=os.environ.get("GITHUB_SHA", "local"))
    args = parser.parse_args()
    try:
        if args.action == "matrix":
            print(json.dumps({"include": targets()}))
        else:
            build(args)
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
