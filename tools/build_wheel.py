"""Assemble a Windows wheel from a prepared native extension."""

import argparse
import hashlib
import os
import re
import tomllib
import zipfile
from pathlib import Path

import lief
from native_abi import load_profile
from wheel.wheelfile import WheelError, WheelFile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "torch_directml"


def validate_native(data):
    binary = lief.PE.parse(data)
    if binary is None:
        raise ValueError("Native input is not a valid PE file")
    if binary.header.machine != lief.PE.Header.MACHINE_TYPES.AMD64:
        raise ValueError("Native input must target AMD64")
    if binary.optional_header.magic != lief.PE.PE_TYPE.PE32_PLUS:
        raise ValueError("Native input must be a 64-bit PE")
    if not binary.header.has_characteristic(lief.PE.Header.CHARACTERISTICS.DLL):
        raise ValueError("Native input must be a PE DLL")
    if not binary.has_exports or "PyInit_torch_directml_native" not in {
        entry.name for entry in binary.get_export().entries if entry.name
    }:
        raise ValueError("Native input lacks PyInit_torch_directml_native export")
    python_libraries = [
        library.name.lower()
        for library in binary.imports
        if library.name.lower().startswith("python")
    ]
    if len(python_libraries) != 1 or not re.fullmatch(
        r"python3\d+\.dll", python_libraries[0]
    ):
        raise ValueError("Native input must link to one ordinary CPython runtime")
    abi = "cp" + python_libraries[0].removeprefix("python").removesuffix(".dll")
    profile = load_profile(abi)
    return binary, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native", type=Path, required=True, help="Prepared CPython .pyd"
    )
    parser.add_argument(
        "--directml-package",
        type=Path,
        required=True,
        help="Official Microsoft.AI.DirectML NuGet package",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    try:
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = config["project"]
        runtime = config["tool"]["torch-directml"]
        name = re.sub(r"[-_.]+", "_", project["name"]).lower()
        version = project["version"]
        dist_info = f"{name}-{version}.dist-info"
        native = args.native.read_bytes()
        binary, profile = validate_native(native)
        abi = profile["python"]["abi"]
        tag = f"{abi}-{abi}-win_amd64"
        major, minor = map(int, profile["python"]["version"].split("."))
        requires_python = f">={major}.{minor},<{major}.{minor + 1}"
        sources = sorted(PACKAGE.glob("*.py"))
        if not sources:
            raise ValueError(f"No Python sources found in {PACKAGE}")
        payloads = [
            (f"torch_directml/{source.name}", source.read_bytes()) for source in sources
        ]
        # Pin the entire NuGet package and copy its official Windows x64 DLL and notices unchanged.
        with args.directml_package.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != runtime["directml-sha256"]:
                raise ValueError(
                    "DirectML package does not match the pinned "
                    f"Microsoft.AI.DirectML {runtime['directml-version']} package"
                )
            stream.seek(0)
            with zipfile.ZipFile(stream) as package:
                payloads.append(
                    (
                        "torch_directml/DirectML.dll",
                        package.read("bin/x64-win/DirectML.dll"),
                    )
                )
                runtime_licenses = {
                    f"directml/{path}": package.read(path)
                    for path in ("LICENSE.txt", "ThirdPartyNotices.txt")
                }
        # Keep the extension outside the package to preserve upstream's absolute import.
        payloads.append((f"torch_directml_native.{abi}-win_amd64.pyd", native))
        if "torch_directml_compat.dll" in {
            library.name.lower() for library in binary.imports
        }:
            raise ValueError(
                "Native input must not depend on torch_directml_compat.dll"
            )
        version_export = next(
            (
                entry
                for entry in binary.get_export().entries
                if entry.name == "tdml_torch_version"
            ),
            None,
        )
        if version_export is None:
            raise ValueError(
                "Native input does not identify its target PyTorch version"
            )
        torch_version = (
            bytes(binary.get_content_from_virtual_address(version_export.address, 64))
            .split(b"\0", 1)[0]
            .decode("ascii")
        )
        if torch_version not in profile["target"]["wheels"]:
            raise ValueError(
                f"Native input identifies unsupported PyTorch {torch_version}"
            )
        # Pin the built wheel to its compiled PyTorch version, independently of the development default.
        dependencies = [
            f"torch=={torch_version}"
            if dependency.startswith("torch==")
            else dependency
            for dependency in project.get("dependencies", [])
        ]
        output_dir = (
            args.output_dir or Path("dist") / f"{abi}-torch{torch_version}-win_amd64"
        )
        output = output_dir / f"{name}-{version}-{torch_version}-{tag}.whl"
        license_files = {
            path: (ROOT / path).read_bytes() for path in project["license-files"]
        }
        license_files.update(runtime_licenses)
        payloads.extend(
            (f"{dist_info}/licenses/{path}", data)
            for path, data in license_files.items()
        )
        readme = (ROOT / project["readme"]).read_text(encoding="utf-8").rstrip("\n")
        headers = [
            "Metadata-Version: 2.4",
            f"Name: {project['name']}",
            f"Version: {version}",
            f"Summary: {project['description']}",
            *(
                f"Project-URL: {label}, {url}"
                for label, url in project.get("urls", {}).items()
            ),
            f"Requires-Python: {requires_python}",
            f"License-Expression: {project['license']}",
            *(f"License-File: {path}" for path in license_files),
            *(f"Requires-Dist: {dependency}" for dependency in dependencies),
            "Description-Content-Type: text/markdown",
            "",
            readme,
            "",
        ]
        metadata = "\n".join(headers)
        wheel_metadata = (
            "Wheel-Version: 1.0\n"
            f"Generator: {project['name']} build_wheel.py\n"
            "Root-Is-Purelib: false\n"
            f"Build: {torch_version}\n"
            f"Tag: {tag}\n"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        # Reserve the output exclusively so wheel creation cannot overwrite an existing file.
        descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.close(descriptor)
        try:
            with WheelFile(output, "w") as wheel:
                for path, data in payloads:
                    wheel.writestr(path, data)
                wheel.writestr(f"{dist_info}/METADATA", metadata)
                wheel.writestr(f"{dist_info}/WHEEL", wheel_metadata)
                wheel.writestr(
                    f"{dist_info}/top_level.txt",
                    "torch_directml\ntorch_directml_native\n",
                )
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        print(output)
    except (KeyError, OSError, ValueError, WheelError, zipfile.BadZipFile) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
