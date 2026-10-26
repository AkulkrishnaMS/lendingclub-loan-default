import argparse


def get_args(desc: str):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", default=None, help="path to CSV/XLSX (default: auto-detect in data/)")
    p.add_argument("--years", nargs=2, type=int, default=None, metavar=("START", "END"),
                   help="keep loans issued in this year range, e.g. --years 2012 2015 (as in the paper)")
    return p.parse_args()
