"""Compile native replacement code for insertion into the upstream extension."""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

import lief
from native_abi import load_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--python-sdk", type=Path, help="Official CPython SDK NuGet package"
    )
    parser.add_argument(
        "--windows-sdk", type=Path, help="xwin splat directory on Linux"
    )
    parser.add_argument("--compiler", default="clang-cl")
    parser.add_argument("--linker", default="lld-link")
    args = parser.parse_args()

    try:
        binary, _, profile = load_inputs(args.source, args.target, args.python_sdk)
        output = (
            args.output_dir or Path(".work") / profile["target"]["id"] / "build"
        ).resolve()
        output.mkdir(parents=True, exist_ok=True)
        torch = output / "torch"
        # Preserve the wheel layout and avoid mixing headers from an installed PyTorch.
        with zipfile.ZipFile(args.target) as wheel:
            for name in wheel.namelist():
                if name.endswith("/"):
                    continue
                if name.startswith("torch/include/") or name in (
                    "torch/lib/c10.lib",
                    "torch/lib/torch_cpu.lib",
                ):
                    destination = output / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with wheel.open(name) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)

        compile_args = [
            args.compiler,
            "/nologo",
            "/std:c++20",
            "/MD",
            "/EHsc",
            "/O2",
            "/DNDEBUG",
            f"/DTDML_SOURCE_ENTRY_RVA={binary.optional_header.addressof_entrypoint:#x}",
            "/clang:-fms-compatibility-version=19.41",
            f"/I{torch / 'include'}",
            f"/I{torch / 'include/torch/csrc/api/include'}",
        ]
        libraries = [f"/LIBPATH:{torch / 'lib'}"]
        if args.python_sdk:
            with zipfile.ZipFile(args.python_sdk) as archive:
                for name in archive.namelist():
                    if name.startswith(
                        ("tools/include/", "tools/libs/")
                    ) and not name.endswith("/"):
                        destination = output / "cpython" / name.removeprefix("tools/")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(archive.read(name))
            compile_args.append(f"/I{output / 'cpython/include'}")
            libraries.append(f"/LIBPATH:{output / 'cpython/libs'}")
        if args.windows_sdk:
            sdk = args.windows_sdk.resolve()
            compile_args.extend(
                f"/imsvc{sdk / path}"
                for path in (
                    "crt/include",
                    "sdk/include/ucrt",
                    "sdk/include/shared",
                    "sdk/include/um",
                    "sdk/include/winrt",
                )
            )
            libraries.extend(
                f"/LIBPATH:{sdk / path}"
                for path in (
                    "crt/lib/x86_64",
                    "sdk/lib/um/x86_64",
                    "sdk/lib/ucrt/x86_64",
                )
            )

        # Reserve the original extension's address range and link the added code after it.
        padding = output / "padding.s"
        reserved = (
            binary.optional_header.sizeof_image
            - binary.optional_header.section_alignment
        )
        padding.write_text(
            '.section .text,"xr"\n.globl tdml_padding\ntdml_padding:\n'
            f".zero {reserved}\n",
            encoding="ascii",
        )
        padding_object = output / "padding.obj"
        subprocess.run(
            [args.compiler, "/nologo", "/c", str(padding), f"/Fo{padding_object}"],
            check=True,
        )
        objects = [str(padding_object)]
        for source in profile["native"]["sources"]:
            obj = output / Path(source).with_suffix(".obj")
            subprocess.run(
                compile_args + ["/c", str(ROOT / "native" / source), f"/Fo{obj}"],
                check=True,
            )
            compiled = lief.COFF.parse(obj)
            if compiled is None:
                raise ValueError(f"Cannot parse the compiled object: {source}")
            if any(section.name.startswith(".CRT$XC") for section in compiled.sections):
                raise ValueError(
                    f"Native code needs an unsupported global initializer: {source}"
                )
            objects.append(str(obj))
        code = output / "native-code.bin"
        subprocess.run(
            [
                args.linker,
                "/DLL",
                "/NOENTRY",
                f"/BASE:{binary.optional_header.imagebase:#x}",
                "/INCLUDE:tdml_padding",
                f"/OUT:{code}",
                *libraries,
                *objects,
                "c10.lib",
                "torch_cpu.lib",
                "kernel32.lib",
                "ucrt.lib",
                "vcruntime.lib",
            ],
            check=True,
        )
        print(code)
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
