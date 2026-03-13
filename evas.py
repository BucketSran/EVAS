#!/usr/bin/env python3
"""EVAS — Event-driven Verilog-A Simulator"""
import argparse
import sys
from evas.netlist.runner import run_spectre


def main():
    parser = argparse.ArgumentParser(
        description='EVAS: Event-driven Verilog-A Simulator')
    parser.add_argument('input', help='.scs netlist file')
    parser.add_argument('-log', help='Log file path')
    parser.add_argument('-o', '--output', default='./output',
                        help='Output directory (default: ./output)')
    args = parser.parse_args()

    ok = run_spectre(args.input, log_path=args.log, output_dir=args.output)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
