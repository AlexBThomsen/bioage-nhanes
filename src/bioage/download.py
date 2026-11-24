"""Download NHANES 1999-2018 tables and the 2019 public-use linked mortality files.

Usage:  python -m bioage.download
Files land in data/raw/<cycle>/ and are only downloaded once.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

XPT_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
MORT_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
    "NHANES_{cycle}_MORT_2019_PUBLIC.dat"
)

# cycle start year -> file suffix
CYCLES = {
    1999: "", 2001: "_B", 2003: "_C", 2005: "_D", 2007: "_E",
    2009: "_F", 2011: "_G", 2013: "_H", 2015: "_I", 2017: "_J",
}

# Logical table -> file stem per cycle. Lab files were renamed in 2005.
def table_names(year: int) -> dict[str, list[str]]:
    s = CYCLES[year]
    early = {1999: ("LAB", ""), 2001: ("L", "_B"), 2003: ("L", "_C")}
    if year in early:
        p, sfx = early[year]
        lab = lambda num: f"{p}{num}{sfx}"  # noqa: E731
        t = {
            "CBC": [lab(25)],
            "BIOPRO": [lab(18) if year == 1999 else lab(40)],
            "CRP": [lab(11)],
            "CHOL": [lab(13)],          # total chol + HDL
            "GHB": [lab(10)],
        }
    else:
        t = {
            "CBC": [f"CBC{s}"],
            "BIOPRO": [f"BIOPRO{s}"],
            "CRP": [f"CRP{s}"] if year <= 2009 else ([f"HSCRP{s}"] if year >= 2015 else []),
            "CHOL": [f"TCHOL{s}"],
            "HDL": [f"HDL{s}"],
            "GHB": [f"GHB{s}"],
        }
    t.update({
        "DEMO": [f"DEMO{s}"],
        "BPX": [f"BPX{s}"] + ([f"BPXO{s}"] if year == 2017 else []),
        "BPQ": [f"BPQ{s}"],
        "BMX": [f"BMX{s}"],
        "SMQ": [f"SMQ{s}"],
        "MCQ": [f"MCQ{s}"],
        "DIQ": [f"DIQ{s}"],
    })
    return t


def fetch(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return f"cached  {dest.name}"
    r = requests.get(url, timeout=120)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or "html" in ctype:
        return f"MISSING {url}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return f"ok      {dest.name} ({len(r.content) // 1024} KB)"


def jobs():
    for year in CYCLES:
        cyc = f"{year}_{year + 1}"
        for stems in table_names(year).values():
            for stem in stems:
                yield XPT_URL.format(year=year, name=stem), RAW / cyc / f"{stem}.xpt"
        yield MORT_URL.format(cycle=cyc), RAW / cyc / f"MORT_{cyc}.dat"


def main() -> int:
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda j: fetch(*j), jobs()))
    for line in results:
        print(line)
    missing = [l for l in results if l.startswith("MISSING")]
    print(f"\n{len(results) - len(missing)} files ok, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
