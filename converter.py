from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .filterbank import FilterbankHeader, FilterbankWriter
from .psrfits import ObservationMetadata, PSRFITSReader


class ConversionError(RuntimeError):
    """Raised when PSRFITS data cannot be converted to filterbank output."""


def convert_psrfits_to_filterbank(
    input_paths: Sequence[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    *,
    overwrite: bool = False,
) -> Path:
    reader = PSRFITSReader(input_paths)
    metadata = reader.metadata

    destination = _resolve_output_path(output_path, metadata.input_paths[0])
    if destination.exists() and not overwrite:
        raise ConversionError(
            f"Output file already exists: {destination}. Use --overwrite to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        FilterbankWriter(handle).write_header(_build_filterbank_header(metadata))
        for subint in reader.iter_subints():
            handle.write(extract_filterbank_block(subint.data, npol=metadata.npol, nsblk=metadata.nsblk))

    return destination


def extract_filterbank_block(
    cell: np.ndarray,
    *,
    npol: int,
    nsblk: int,
) -> bytes:
    normalized = _normalize_data_cell(cell, npol=npol, nsblk=nsblk)
    contiguous = np.ascontiguousarray(normalized)
    if contiguous.dtype.itemsize == 1:
        return contiguous.tobytes()
    return contiguous.astype(contiguous.dtype.newbyteorder("<"), copy=False).tobytes()


def _normalize_data_cell(
    cell: np.ndarray,
    *,
    npol: int,
    nsblk: int,
) -> np.ndarray:
    array = np.asarray(cell)
    while array.ndim > 0 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim == 1:
        if array.size % nsblk != 0:
            raise ConversionError("Could not infer the DATA cell layout from a flat array.")
        width = array.size // nsblk
        if width % npol != 0:
            raise ConversionError("Flat DATA array is not divisible by NPOL.")
        array = array.reshape(nsblk, npol, width // npol)
    elif array.ndim == 2:
        if array.shape[0] == nsblk and npol == 1:
            array = array.reshape(nsblk, 1, array.shape[1])
        elif array.shape[0] == npol and nsblk == 1:
            array = array.reshape(1, npol, array.shape[1])
        else:
            raise ConversionError(f"Unsupported 2D DATA cell shape {array.shape}.")
    elif array.ndim != 3:
        raise ConversionError(f"Unsupported DATA cell shape {array.shape}.")

    if array.shape[0] != nsblk or array.shape[1] != npol:
        raise ConversionError(
            f"DATA cell shape {array.shape} does not match NSBLK={nsblk}, NPOL={npol}."
        )

    return np.ascontiguousarray(array)


def _resolve_output_path(output_path: Optional[Union[str, Path]], first_input_path: Path) -> Path:
    if output_path is None:
        return first_input_path.with_suffix(".fil")
    return Path(output_path).expanduser().resolve()


def _build_filterbank_header(metadata: ObservationMetadata) -> FilterbankHeader:
    fch1_mhz, foff_mhz = _build_frequency_axis(metadata)
    return FilterbankHeader(
        rawdatafile=metadata.rawdatafile,
        source_name=metadata.source_name,
        data_type=1,
        nchans=metadata.nchan,
        fch1=fch1_mhz,
        foff=foff_mhz,
        nbits=metadata.nbits,
        nifs=metadata.npol,
        tsamp=metadata.tsamp_seconds,
        tstart=metadata.tstart_mjd,
        src_raj=_sigproc_ra(metadata.ra_str),
        src_dej=_sigproc_dec(metadata.dec_str),
        az_start=metadata.first_az_deg,
        za_start=metadata.first_za_deg,
    )


def _build_frequency_axis(metadata: ObservationMetadata) -> Tuple[float, float]:
    freqs = None
    if metadata.first_row_freqs_mhz is not None:
        freqs = np.asarray(metadata.first_row_freqs_mhz, dtype=np.float64)
        if freqs.size != metadata.nchan:
            freqs = None

    default_step = metadata.chan_bw_mhz
    if default_step == 0.0 and metadata.nchan:
        default_step = metadata.bandwidth_mhz / metadata.nchan
    if default_step == 0.0:
        default_step = -1.0

    if freqs is not None and freqs.size >= 2:
        return float(freqs[0]), float(np.median(np.diff(freqs)))

    fch1_mhz = metadata.center_freq_mhz - default_step * (metadata.nchan - 1) / 2.0
    return fch1_mhz, float(default_step)


def _sigproc_ra(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    hours, minutes, seconds = (float(part) for part in parts)
    return hours * 10000.0 + minutes * 100.0 + seconds


def _sigproc_dec(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    sign = -1.0 if degrees < 0 else 1.0
    return sign * (abs(degrees) * 10000.0 + minutes * 100.0 + seconds)
