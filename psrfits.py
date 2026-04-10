from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Set, Tuple, Union

from astropy.io import fits
import numpy as np


class PSRFITSError(RuntimeError):
    """Raised when the input PSRFITS files are unsupported or inconsistent."""


@dataclass(frozen=True)
class ObservationMetadata:
    input_paths: Tuple[Path, ...]
    rawdatafile: str
    source_name: str
    ra_str: Optional[str]
    dec_str: Optional[str]
    nchan: int
    npol: int
    nbits: int
    nsblk: int
    tstart_mjd: float
    tsamp_seconds: float
    bandwidth_mhz: float
    chan_bw_mhz: float
    center_freq_mhz: float
    first_row_freqs_mhz: Optional[Tuple[float, ...]]
    first_az_deg: Optional[float]
    first_za_deg: Optional[float]


@dataclass(frozen=True)
class SubintRecord:
    data: np.ndarray


class PSRFITSReader:
    def __init__(self, input_paths: Sequence[Union[str, Path]]) -> None:
        if not input_paths:
            raise PSRFITSError("At least one PSRFITS input file is required.")

        self._paths = tuple(Path(path).expanduser().resolve() for path in input_paths)
        missing = [str(path) for path in self._paths if not path.exists()]
        if missing:
            raise PSRFITSError(f"Input file not found: {missing[0]}")

        self.metadata = self._read_metadata(self._paths[0])

    def iter_subints(self) -> Iterator[SubintRecord]:
        for path in self._paths:
            with fits.open(path, mode="readonly", memmap=True) as hdul:
                primary_header = hdul[0].header
                subint_hdu = hdul["SUBINT"]
                self._validate_consistency(path, primary_header, subint_hdu.header)

                table_data = subint_hdu.data
                if table_data is None:
                    continue

                for row in table_data:
                    yield SubintRecord(data=np.asarray(row["DATA"]))

    def _read_metadata(self, path: Path) -> ObservationMetadata:
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            primary_header = hdul[0].header
            subint_hdu = hdul["SUBINT"]
            subint_header = subint_hdu.header
            table_data = subint_hdu.data

            if table_data is None or len(table_data) == 0:
                raise PSRFITSError(f"SUBINT table is empty in {path}")

            obs_mode = str(primary_header.get("OBS_MODE", "")).strip().upper()
            if not obs_mode.startswith("SEARCH"):
                raise PSRFITSError(
                    f"Only SEARCH mode PSRFITS is supported, found '{obs_mode or 'UNKNOWN'}'."
                )

            first_row = table_data[0]
            available_names = set(table_data.names or [])

            return ObservationMetadata(
                input_paths=self._paths,
                rawdatafile=path.name,
                source_name=str(primary_header.get("SRC_NAME", "UNKNOWN")).strip() or "UNKNOWN",
                ra_str=_optional_header_string(primary_header, "RA"),
                dec_str=_optional_header_string(primary_header, "DEC"),
                nchan=int(subint_header["NCHAN"]),
                npol=int(subint_header["NPOL"]),
                nbits=int(subint_header["NBITS"]),
                nsblk=int(subint_header["NSBLK"]),
                tstart_mjd=(
                    float(primary_header.get("STT_IMJD", 0.0))
                    + float(primary_header.get("STT_SMJD", 0.0)) / 86400.0
                    + float(primary_header.get("STT_OFFS", 0.0)) / 86400.0
                ),
                tsamp_seconds=float(subint_header["TBIN"]),
                bandwidth_mhz=float(primary_header.get("OBSBW", 0.0)),
                chan_bw_mhz=float(subint_header.get("CHAN_BW", 0.0)),
                center_freq_mhz=float(primary_header.get("OBSFREQ", 0.0)),
                first_row_freqs_mhz=_optional_freqs(first_row, available_names),
                first_az_deg=_optional_scalar(first_row, available_names, "TEL_AZ"),
                first_za_deg=_optional_scalar(first_row, available_names, "TEL_ZEN"),
            )

    def _validate_consistency(
        self,
        path: Path,
        primary_header: fits.Header,
        subint_header: fits.Header,
    ) -> None:
        obs_mode = str(primary_header.get("OBS_MODE", "")).strip().upper()
        if not obs_mode.startswith("SEARCH"):
            raise PSRFITSError(f"{path.name} is not a SEARCH mode PSRFITS file.")

        metadata = self.metadata
        for key, expected in {
            "NCHAN": metadata.nchan,
            "NPOL": metadata.npol,
            "NBITS": metadata.nbits,
            "NSBLK": metadata.nsblk,
        }.items():
            actual = int(subint_header[key])
            if actual != expected:
                raise PSRFITSError(
                    f"{path.name} has {key}={actual}, expected {expected} from the first file."
                )


def _optional_header_string(header: fits.Header, name: str) -> Optional[str]:
    value = header.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_scalar(row: fits.FITS_record, names: Set[str], name: str) -> Optional[float]:
    if name not in names:
        return None
    return float(row[name])


def _optional_freqs(row: fits.FITS_record, names: Set[str]) -> Optional[Tuple[float, ...]]:
    if "DAT_FREQ" not in names:
        return None
    array = np.asarray(row["DAT_FREQ"], dtype=np.float64).reshape(-1)
    return tuple(float(value) for value in array)
