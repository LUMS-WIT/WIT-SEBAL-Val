# Original batch runner authors: tih and Sajid Pareeth (2019).
"""Standalone central-product model runner; never called by main.py."""
from pathlib import Path
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--excel', type=Path, required=True)
    parser.add_argument('--start-row', type=int, default=2)
    parser.add_argument('--end-row', type=int, required=True)
    args = parser.parse_args(argv)
    if args.start_row < 2 or args.end_row < args.start_row:
        parser.error('Provide an increasing inclusive workbook row range starting at 2 or later.')
    if not args.excel.is_file():
        parser.error(f'Workbook not found: {args.excel}')
    import pysebal_py3
    # Calibration constants in the original model must match the intended product.
    for row in range(args.start_row, args.end_row + 1):
        print(f'Running workbook row {row}', flush=True)
        pysebal_py3.SEBALcode(row, str(args.excel))


if __name__ == '__main__':
    main()
