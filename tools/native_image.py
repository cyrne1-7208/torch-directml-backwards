"""Place compiled native code in the original extension's address space."""

import struct

import lief


def append_native_code(binary, code):
    if code.has_tls or code.optional_header.addressof_entrypoint:
        raise ValueError("Native code must have no separate TLS or CRT entry point")
    if code.optional_header.imagebase != binary.optional_header.imagebase:
        raise ValueError("Native code was linked for a different image base")
    end = binary.optional_header.sizeof_image
    symbols = {entry.name: entry.address for entry in code.get_export().entries}
    if any(address < end for address in symbols.values()):
        raise ValueError("Native code overlaps the original extension")

    for source in code.sections:
        start = source.virtual_address
        content = bytes(source.content)
        virtual_size = source.virtual_size
        if start < end:
            skip = end - start
            if source.name != ".text" or len(content) < skip or any(content[:skip]):
                raise ValueError("The reserved source-image range contains native code")
            start += skip
            content = content[skip:]
            virtual_size -= skip
        target = lief.PE.Section(".nt" + source.name[1:])
        target.content = list(content)
        target.virtual_address = start
        target.virtual_size = virtual_size
        target.characteristics = source.characteristics
        added = binary.add_section(target)
        if added.virtual_address != start:
            raise ValueError("Native section addresses changed while appending code")

    # Preserve linked RVAs so C++ type and exception metadata need no individual address fixups.
    functions = []
    for image in (binary, code):
        directory = image.data_directory(lief.PE.DataDirectory.TYPES.EXCEPTION_TABLE)
        data = bytes(
            image.get_content_from_virtual_address(directory.rva, directory.size)
        )
        functions.extend(struct.iter_unpack("<III", data))
    functions.sort()
    metadata = bytearray().join(
        struct.pack("<III", *function) for function in functions
    )
    relocation_offset = len(metadata)
    pages = {}
    for image in (binary, code):
        for block in image.relocations:
            for entry in block.entries:
                if entry.type != lief.PE.RelocationEntry.BASE_TYPES.ABS:
                    pages.setdefault(block.virtual_address, []).append(entry.data)
    for page, entries in sorted(pages.items()):
        if len(entries) % 2:
            entries.append(0)
        metadata.extend(struct.pack("<II", page, 8 + len(entries) * 2))
        metadata.extend(struct.pack(f"<{len(entries)}H", *entries))
    section = lief.PE.Section(".ntmeta")
    section.content = list(metadata)
    section.characteristics = int(
        lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA
        | lief.PE.Section.CHARACTERISTICS.MEM_READ
    )
    section = binary.add_section(section)
    directory = binary.data_directory(lief.PE.DataDirectory.TYPES.EXCEPTION_TABLE)
    directory.rva = section.virtual_address
    directory.size = relocation_offset
    directory = binary.data_directory(lief.PE.DataDirectory.TYPES.BASE_RELOCATION_TABLE)
    directory.rva = section.virtual_address + relocation_offset
    directory.size = len(metadata) - relocation_offset
    binary.optional_header.addressof_entrypoint = symbols["tdml_entry"]
    return symbols
