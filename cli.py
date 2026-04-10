from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .converter import ConversionError, convert_psrfits_to_filterbank
from .psrfits import PSRFITSError


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fits2fil",
        description="Convert PSRFITS SEARCH data into SIGPROC filterbank format.",
    )
    parser.add_argument("inputs", nargs="+", help="Input PSRFITS files.")
    parser.add_argument("-o", "--output", help="Output .fil path. Defaults to the first input name.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    args = parser.parse_args(argv)

    input_paths = [Path(item).expanduser().resolve() for item in args.inputs]

    try:
        result = convert_psrfits_to_filterbank(
            input_paths,
            args.output,
            overwrite=args.overwrite,
        )
    except (ConversionError, PSRFITSError) as exc:
        parser.exit(status=2, message=f"fits2fil: error: {exc}\n")

    print(f"Wrote {result}")
    return 0
