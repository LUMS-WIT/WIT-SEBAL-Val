# Original batch runner authors: tih and Sajid Pareeth (2019).
"""Standalone central-product model runner; never called by main.py."""
from pathlib import Path
import argparse
import math


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--excel', type=Path, required=True)
    parser.add_argument('--start-row', type=int, default=2)
    parser.add_argument('--end-row', type=int, required=True)
    parser.add_argument('--alpha', type=float, default=-.089,
                        help='EF slope for the study central product (default: -0.089).')
    parser.add_argument('--beta', type=float, default=.343,
                        help='EF intercept for the study central product (default: 0.343).')
    args = parser.parse_args(argv)
    if args.start_row < 2 or args.end_row < args.start_row:
        parser.error('Provide an increasing inclusive workbook row range starting at 2 or later.')
    if not args.excel.is_file():
        parser.error(f'Workbook not found: {args.excel}')
    if not all(math.isfinite(value) for value in (args.alpha, args.beta)):
        parser.error('Calibration coefficients must be finite.')
    import pysebal_py3
    # Explicitly override the legacy lower-member constants in the model source.
    pysebal_py3.ALPHA, pysebal_py3.BETA = args.alpha, args.beta
    pysebal_py3.CALIBRATED, pysebal_py3.CALIBRATIONS = True, False
    print(f'Central-product EF calibration: alpha={args.alpha}, beta={args.beta}', flush=True)
    for row in range(args.start_row, args.end_row + 1):
        print(f'Running workbook row {row}', flush=True)
        pysebal_py3.SEBALcode(row, str(args.excel))


if __name__ == '__main__':
    main()
