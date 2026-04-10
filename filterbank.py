from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import BinaryIO, Optional


@dataclass(frozen=True)
class FilterbankHeader:
    rawdatafile: str
    source_name: str
    data_type: int
    nchans: int
    fch1: float
    foff: float
    nbits: int
    nifs: int
    tsamp: float
    tstart: float
    src_raj: Optional[float] = None
    src_dej: Optional[float] = None
    az_start: Optional[float] = None
    za_start: Optional[float] = None


class FilterbankWriter:
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle

    def write_header(self, header: FilterbankHeader) -> None:
        self._write_string("HEADER_START")
        self._write_name_and_string("rawdatafile", header.rawdatafile)
        self._write_name_and_string("source_name", header.source_name)
        self._write_name_and_int("data_type", header.data_type)
        self._write_name_and_int("nchans", header.nchans)
        self._write_name_and_double("fch1", header.fch1)
        self._write_name_and_double("foff", header.foff)
        self._write_name_and_int("nbits", header.nbits)
        self._write_name_and_int("nifs", header.nifs)
        self._write_name_and_double("tsamp", header.tsamp)
        self._write_name_and_double("tstart", header.tstart)
        self._write_optional_double("src_raj", header.src_raj)
        self._write_optional_double("src_dej", header.src_dej)
        self._write_optional_double("az_start", header.az_start)
        self._write_optional_double("za_start", header.za_start)
        self._write_string("HEADER_END")

    def _write_optional_double(self, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        self._write_name_and_double(name, value)

    def _write_name_and_string(self, name: str, value: str) -> None:
        self._write_string(name)
        self._write_string(value)

    def _write_name_and_int(self, name: str, value: int) -> None:
        self._write_string(name)
        self.handle.write(struct.pack("<i", value))

    def _write_name_and_double(self, name: str, value: float) -> None:
        self._write_string(name)
        self.handle.write(struct.pack("<d", value))

    def _write_string(self, value: str) -> None:
        encoded = value.encode("ascii", errors="replace")
        self.handle.write(struct.pack("<i", len(encoded)))
        self.handle.write(encoded)
