"""Download the OWID COVID-19 dataset into data/raw/.

The file is about 100 MB, so it is not committed to git. Run this once:

    python data/download_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
DEST = Path(__file__).resolve().parent / "raw" / "owid-covid-data.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urlretrieve(URL, DEST)
    print(f"Saved to {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
