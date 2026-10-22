"""Little-endian binary reader with Frostbite/.NET conventions.

GUIDs are stored the way .NET's ``new Guid(byte[16])`` reads them (first three
groups little-endian), which is also how Frosty prints chunk ids in .fbmod
resource names. ``read_guid`` returns that canonical lowercase string.
"""

from __future__ import annotations

import struct
import uuid


class FormatError(Exception):
    """Raised when binary data does not match the expected structure."""


class Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def __len__(self) -> int:
        return len(self.data)

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def seek(self, pos: int) -> None:
        if pos < 0 or pos > len(self.data):
            raise FormatError(f"seek to 0x{pos:x} outside buffer of 0x{len(self.data):x} bytes")
        self.pos = pos

    def skip(self, n: int) -> None:
        self.seek(self.pos + n)

    def align(self, alignment: int) -> None:
        if alignment > 1 and self.pos % alignment:
            self.pos += alignment - (self.pos % alignment)

    def _unpack(self, fmt: str, size: int):
        if self.pos + size > len(self.data):
            raise FormatError(
                f"read of {size} bytes at 0x{self.pos:x} overruns buffer of 0x{len(self.data):x} bytes"
            )
        value = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return value

    def u8(self) -> int:
        return self._unpack("<B", 1)[0]

    def i8(self) -> int:
        return self._unpack("<b", 1)[0]

    def u16(self) -> int:
        return self._unpack("<H", 2)[0]

    def i16(self) -> int:
        return self._unpack("<h", 2)[0]

    def u32(self) -> int:
        return self._unpack("<I", 4)[0]

    def i32(self) -> int:
        return self._unpack("<i", 4)[0]

    def u64(self) -> int:
        return self._unpack("<Q", 8)[0]

    def i64(self) -> int:
        return self._unpack("<q", 8)[0]

    def f32(self) -> float:
        return self._unpack("<f", 4)[0]

    def f64(self) -> float:
        return self._unpack("<d", 8)[0]

    def u32_be(self) -> int:
        return self._unpack(">I", 4)[0]

    def u16_be(self) -> int:
        return self._unpack(">H", 2)[0]

    def bytes(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise FormatError(f"read of {n} bytes at 0x{self.pos:x} overruns buffer of 0x{len(self.data):x} bytes")
        value = bytes(self.data[self.pos:self.pos + n])
        self.pos += n
        return value

    def guid(self) -> str:
        return guid_from_bytes(self.bytes(16))

    def sha1(self) -> str:
        return self.bytes(20).hex()

    def cstring(self) -> str:
        end = self.data.find(b"\x00", self.pos)
        if end < 0:
            raise FormatError(f"unterminated string at 0x{self.pos:x}")
        raw = self.data[self.pos:end]
        self.pos = end + 1
        return decode_text(raw)

    def sized_string(self, n: int) -> str:
        return decode_text(self.bytes(n))

    def fixed_string(self, n: int) -> str:
        raw = self.bytes(n)
        return decode_text(raw.split(b"\x00", 1)[0])

    def cstring_at(self, offset: int) -> str:
        saved = self.pos
        try:
            self.seek(offset)
            return self.cstring()
        finally:
            self.pos = saved


def decode_text(raw: bytes) -> str:
    # Frosty's NativeWriter writes each UTF-16 char as one byte, so text is
    # effectively Latin-1; UTF-8 is tried first for mods written by other tools.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def guid_from_bytes(raw: bytes) -> str:
    return str(uuid.UUID(bytes_le=bytes(raw)))


def guid_to_bytes(guid: str) -> bytes:
    return uuid.UUID(guid).bytes_le


NULL_GUID = "00000000-0000-0000-0000-000000000000"
