"""Inspect and apply reviewed edits to the upstream Windows extension."""

import argparse
import hashlib
import json
import struct
import tomllib
import zipfile
from pathlib import Path

import capstone
import lief
from native_image import append_native_code

PROFILES = Path(__file__).resolve().parents[1] / "native" / "profiles"


def load_profile(abi):
    profile = tomllib.loads((PROFILES / "common.toml").read_text(encoding="utf-8"))
    specific = tomllib.loads((PROFILES / f"{abi}.toml").read_text(encoding="utf-8"))
    profile["python"] = specific["python"] | {
        "abi": abi,
        "library": f"python{abi[2:]}.dll",
    }
    profile["target"]["wheels"] = specific["target"]["wheels"]
    for field in ("imports", "edits"):
        profile[field].extend(specific.get(field, []))
    for field in ("sources", "trampolines"):
        profile["native"][field].extend(specific.get("native", {}).get(field, []))
    return profile


def parse_pe(data):
    binary = lief.PE.parse(data)
    if binary is None:
        raise ValueError("Cannot parse the PE file")
    return binary


def load_inputs(source, target, python_sdk=None):
    _, python_tag, abi, platform = target.stem.rsplit("-", 3)
    if (
        python_tag != abi
        or platform != "win_amd64"
        or not abi.startswith("cp3")
        or not abi[2:].isdigit()
    ):
        raise ValueError(f"Unsupported target wheel tags: {target.name}")
    profile = load_profile(abi)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != profile["source"]["sha256"]:
        raise ValueError(f"Unexpected wheel: {source}; SHA-256 is {digest}")
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    version = next(
        (
            name
            for name, value in profile["target"]["wheels"].items()
            if value == digest
        ),
        None,
    )
    if version is None:
        raise ValueError(f"Unexpected wheel: {target}; SHA-256 is {digest}")
    profile["target"]["torch_version"] = version
    profile["target"]["version"] = (
        f"torch {version} / CPython {profile['python']['version']} / Windows x64"
    )
    profile["target"]["id"] = f"{abi}-torch{version}-win_amd64"
    with zipfile.ZipFile(source) as wheel:
        binary = parse_pe(wheel.read(profile["source"]["member"]))
    exports = {}
    with zipfile.ZipFile(target) as wheel:
        for library in profile["target"]["libraries"]:
            dependency = parse_pe(wheel.read(f"torch/lib/{library}"))
            exports[library] = {
                entry.name for entry in dependency.get_export().entries if entry.name
            }
    if (sdk := profile["python"].get("sdk")) and (
        python_sdk is not None or abi != "cp312"
    ):
        if python_sdk is None:
            raise ValueError(
                f"{abi} requires --python-sdk with the official CPython SDK"
            )
        with python_sdk.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != sdk["sha256"]:
            raise ValueError(
                f"Unexpected CPython SDK: {python_sdk}; SHA-256 is {digest}"
            )
        with zipfile.ZipFile(python_sdk) as archive:
            library = profile["python"]["library"]
            dependency = parse_pe(archive.read(f"tools/{library}"))
            exports[library] = {
                entry.name for entry in dependency.get_export().entries if entry.name
            }
    return binary, exports, profile


def unresolved_imports(binary, exports):
    result = {}
    for library in binary.imports:
        if library.name in exports:
            result.setdefault(library.name, []).extend(
                entry.name
                for entry in library.entries
                if entry.name and entry.name not in exports[library.name]
            )
    return result


def edit_instructions(binary, changes):
    # Edit only reviewed locations in the pinned wheel, not other instructions containing the same values.
    for change in changes:
        before = bytes.fromhex(change["before"])
        after = bytes.fromhex(change["after"])
        if len(before) != len(after):
            raise ValueError(
                f"An in-place edit changes instruction size: {change['reason']}"
            )
        for rva in change["rvas"]:
            for section in binary.sections:
                end = rva + len(before) - section.virtual_address
                if section.virtual_address <= rva and end <= section.size:
                    if end > section.virtual_size:
                        # Use trailing padding in the pinned source wheel for a small leaf function.
                        section.content = list(bytes(section.content).ljust(end, b"\0"))
                        section.virtual_size = end
                    break
            found = bytes(binary.get_content_from_virtual_address(rva, len(before)))
            if found != before:
                raise ValueError(f"Instruction does not match at RVA {rva:#x}")
            binary.patch_address(rva, list(after))


def serialize(binary, rebuild_exports=False):
    config = lief.PE.Builder.config_t()
    config.exports = rebuild_exports
    # Keep the merged relocation table and original TLS instead of rebuilding them through Builder.
    for attribute in ("debug", "load_configuration", "resources", "relocations", "tls"):
        setattr(config, attribute, False)
    result = bytearray(binary.write_to_bytes(config))
    debug = binary.data_directory(lief.PE.DataDirectory.TYPES.DEBUG_DIR)
    if debug.size:
        # Added sections can enlarge the headers; recompute debug data's file offsets from its RVAs.
        pe = struct.unpack_from("<I", result, 0x3C)[0]
        count = struct.unpack_from("<H", result, pe + 6)[0]
        optional_size = struct.unpack_from("<H", result, pe + 20)[0]
        sections = [
            struct.unpack_from(
                "<IIII", result, pe + 24 + optional_size + index * 40 + 8
            )
            for index in range(count)
        ]

        def file_offset(rva):
            return next(
                raw + rva - start
                for _, start, size, raw in sections
                if start <= rva < start + size
            )

        table = file_offset(debug.rva)
        for entry in range(table, table + debug.size, 28):
            address = struct.unpack_from("<I", result, entry + 20)[0]
            if address:
                struct.pack_into("<I", result, entry + 24, file_offset(address))
    return bytes(result)


def rebuild_imports(binary, replacements, native_imports=()):
    groups = {}
    removed = set()
    native_calls = {}
    libraries = [(library, replacements) for library in binary.imports]
    libraries.extend((library, {}) for library in native_imports)
    for library, changes in libraries:
        for entry in library.entries:
            replacement = changes.get(entry.name, (library.name, entry.name))
            if replacement is None:
                removed.add(entry.iat_address)
                continue
            if isinstance(replacement, int):
                native_calls[entry.iat_address] = replacement
                continue
            target, symbol = replacement
            value = (1 << 63) | entry.ordinal if entry.is_ordinal else symbol
            groups.setdefault(target, {}).setdefault(value, []).append(
                entry.iat_address
            )

    descriptor_size = (len(groups) + 1) * 20
    payload = bytearray(descriptor_size)
    payload.extend(b"\0" * (-len(payload) % 8))
    iat_start = len(payload)
    iat_size = sum(len(entries) + 1 for entries in groups.values()) * 8
    payload.extend(b"\0" * iat_size)
    fixups = []
    slots = {}
    iat = iat_start
    for index, (library, entries) in enumerate(groups.items()):
        name = len(payload)
        payload.extend(library.encode("ascii") + b"\0")
        hints = []
        for symbol, original_slots in entries.items():
            for old_slot in original_slots:
                slots[old_slot] = iat + len(hints) * 8
            if isinstance(symbol, int):
                hints.append((symbol, False))
                continue
            if len(payload) % 2:
                payload.append(0)
            hints.append((len(payload), True))
            payload.extend(b"\0\0" + symbol.encode("ascii") + b"\0")
        payload.extend(b"\0" * (-len(payload) % 8))
        lookup = len(payload)
        for number, (value, relative) in enumerate(hints):
            slot = iat + number * 8
            if relative:
                fixups.extend(((len(payload), "<Q", value), (slot, "<Q", value)))
            struct.pack_into("<Q", payload, slot, value)
            payload.extend(struct.pack("<Q", value))
        payload.extend(b"\0" * 8)
        struct.pack_into("<IIIII", payload, index * 20, lookup, 0, 0, name, iat)
        fixups.extend(
            (
                (index * 20, "<I", lookup),
                (index * 20 + 12, "<I", name),
                (index * 20 + 16, "<I", iat),
            )
        )
        iat += (len(entries) + 1) * 8

    section = lief.PE.Section(".tdml")
    section.characteristics = int(
        lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA
        | lief.PE.Section.CHARACTERISTICS.MEM_READ
    )
    section.content = list(payload)
    section = binary.add_section(section)
    for offset, kind, value in fixups:
        struct.pack_into(kind, payload, offset, section.virtual_address + value)
    section.content = list(payload)
    directory = binary.data_directory(lief.PE.DataDirectory.TYPES.IMPORT_TABLE)
    directory.rva = section.virtual_address
    directory.size = descriptor_size
    directory = binary.data_directory(lief.PE.DataDirectory.TYPES.IAT)
    directory.rva = section.virtual_address + iat_start
    directory.size = iat_size
    bound = binary.data_directory(lief.PE.DataDirectory.TYPES.BOUND_IMPORT)
    bound.rva = 0
    bound.size = 0

    slots = {old: section.virtual_address + new for old, new in slots.items()}
    # Build a terminated IAT for each DLL and redirect the existing instruction references.
    decoder = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    decoder.detail = True
    decoder.skipdata = True
    for text in binary.sections:
        if not text.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE):
            continue
        for instruction in decoder.disasm(bytes(text.content), text.virtual_address):
            if instruction.id == 0:
                continue
            for operand in instruction.operands:
                if operand.type != capstone.x86.X86_OP_MEM:
                    continue
                if operand.mem.base != capstone.x86.X86_REG_RIP:
                    continue
                destination = instruction.address + instruction.size + operand.mem.disp
                if destination in native_calls:
                    if (
                        instruction.mnemonic not in ("call", "jmp")
                        or instruction.size < 5
                    ):
                        raise ValueError(
                            f"Unsupported native reference at RVA {instruction.address:#x}"
                        )
                    opcode = b"\xe8" if instruction.mnemonic == "call" else b"\xe9"
                    displacement = native_calls[destination] - instruction.address - 5
                    binary.patch_address(
                        instruction.address,
                        list(
                            opcode
                            + struct.pack("<i", displacement)
                            + b"\x90" * (instruction.size - 5)
                        ),
                    )
                    continue
                if destination in removed:
                    raise ValueError(
                        f"Removed import is still used at RVA {instruction.address:#x}"
                    )
                if destination in slots:
                    displacement = (
                        slots[destination] - instruction.address - instruction.size
                    )
                    binary.patch_address(
                        instruction.address + instruction.disp_offset,
                        list(struct.pack("<i", displacement)),
                    )

    # Update absolute addresses in data using the original base relocations.
    image = binary.optional_header.imagebase
    for relocation in binary.relocations:
        for entry in relocation.entries:
            if entry.type != lief.PE.RelocationEntry.BASE_TYPES.DIR64:
                continue
            address = relocation.virtual_address + entry.position
            value = struct.unpack(
                "<Q", bytes(binary.get_content_from_virtual_address(address, 8))
            )[0]
            if value - image in removed:
                raise ValueError(
                    f"Removed import is still referenced at RVA {address:#x}"
                )
            if value - image in native_calls:
                raise ValueError(
                    f"Unsupported native address reference at RVA {address:#x}"
                )
            if value - image in slots:
                binary.patch_address(
                    address, list(struct.pack("<Q", image + slots[value - image]))
                )


def apply_native(binary, profile, compiled, exports):
    code = parse_pe(compiled.read_bytes())
    if any(
        library.name.lower().startswith("python")
        and library.name.lower() != profile["python"]["library"]
        for library in code.imports
    ):
        raise ValueError("Native code imports a different CPython runtime")
    symbols = {entry.name: entry.address for entry in code.get_export().entries}
    required = set(profile["native"]["overrides"].values())
    required.update(entry["symbol"] for entry in profile["native"]["trampolines"])
    required.update(("tdml_entry", "tdml_torch_version"))
    missing = required - symbols.keys()
    if missing:
        raise ValueError(f"Native code lacks required functions: {sorted(missing)}")
    version = bytes(
        code.get_content_from_virtual_address(symbols["tdml_torch_version"], 64)
    )
    version = version.split(b"\0", 1)[0].decode("ascii")
    if version != profile["target"]["torch_version"]:
        raise ValueError(f"Native code was compiled for PyTorch {version}")
    if any(unresolved_imports(code, exports).values()):
        raise ValueError("Native code imports do not match the selected PyTorch DLLs")
    append_native_code(binary, code)
    # Reload the merged relocations so IAT updates also cover absolute references in the added code.
    binary = parse_pe(serialize(binary))
    edit_instructions(binary, profile["edits"] + profile["native"]["edits"])
    replacements = {
        entry["before"]: (entry["library"], entry["after"])
        for entry in profile["imports"]
    }
    for library in binary.imports:
        for entry in library.entries:
            if library.name.lower() == profile["source"]["python_library"]:
                replacements.setdefault(
                    entry.name, (profile["python"]["library"], entry.name)
                )
            if entry.name in profile["native"]["aliases"]:
                replacements[entry.name] = (
                    library.name,
                    profile["native"]["aliases"][entry.name],
                )
    replacements.update(
        {
            old: symbols[current]
            for old, current in profile["native"]["overrides"].items()
        }
    )
    replacements.update(dict.fromkeys(profile["obsolete"]["imports"]))
    for replacement in profile["native"]["trampolines"]:
        start = replacement["rva"]
        expected = bytes.fromhex(replacement["before"])
        if (
            bytes(binary.get_content_from_virtual_address(start, len(expected)))
            != expected
        ):
            raise ValueError(f"Native replacement does not match at RVA {start:#x}")
        opcode = b"\xe8" if replacement.get("call", False) else b"\xe9"
        branch = opcode + struct.pack("<i", symbols[replacement["symbol"]] - start - 5)
        binary.patch_address(
            start, list(branch + b"\x90" * (len(expected) - len(branch)))
        )
    rebuild_imports(binary, replacements, code.imports)
    exported = binary.get_export()
    ordinal = max(entry.ordinal for entry in exported.entries) + 1
    marker = exported.add_entry("tdml_torch_version", symbols["tdml_torch_version"])
    marker.ordinal = ordinal
    return serialize(binary, rebuild_exports=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "patch"))
    parser.add_argument(
        "source", type=Path, help="Original torch-directml Windows wheel"
    )
    parser.add_argument(
        "target", type=Path, help="Reviewed official PyTorch Windows wheel"
    )
    parser.add_argument("--output", type=Path, help="New patched .pyd file")
    parser.add_argument(
        "--native-code", type=Path, help="Linked native-code.bin from build_native.py"
    )
    parser.add_argument(
        "--python-sdk", type=Path, help="Official CPython SDK NuGet package"
    )
    args = parser.parse_args()
    if args.action == "patch" and (args.output is None or args.native_code is None):
        parser.error("patch requires --output and --native-code")
    if args.action == "inspect" and args.output is not None:
        parser.error("--output is only used with patch")
    if args.native_code and args.action != "patch":
        parser.error("--native-code is only used with patch")

    try:
        binary, exports, profile = load_inputs(
            args.source, args.target, args.python_sdk
        )
        report = {
            "source": profile["source"]["version"],
            "target": profile["target"]["version"],
            "unresolved_imports": unresolved_imports(binary, exports),
            "validation": "Import resolution only; runtime checks are documented in docs/native-abi.md",
        }
        if args.action == "patch":
            result = apply_native(binary, profile, args.native_code, exports)
            patched = parse_pe(result)
            report["unresolved_imports"] = unresolved_imports(patched, exports)
            if any(report["unresolved_imports"].values()):
                raise ValueError(
                    f"Unresolved target imports: {report['unresolved_imports']}"
                )
            # Never overwrite source files or existing build outputs.
            with args.output.open("xb") as stream:
                stream.write(result)
            report["output"] = str(args.output)
            report["sha256"] = hashlib.sha256(result).hexdigest()
            report["status"] = "Patched native candidate; runtime validation required"
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
