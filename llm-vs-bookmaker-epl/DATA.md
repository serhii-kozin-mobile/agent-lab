# Data: not redistributed here

This study uses Premier League results and bookmaker odds from **Football-Data.co.uk**:
https://www.football-data.co.uk/englandm.php (file key: https://www.football-data.co.uk/notes.txt)

Seasons used: 2021-22 to 2025-26 (`mmz4281/<season>/E0.csv`).

The CSVs are **not included** in this repository, and neither is `out/matches_2526.json`
(it is built from them and contains the odds). Football-Data.co.uk publishes the files for free
but does not publish an explicit redistribution licence, so this repo links to the source
instead of copying it. Please download from the source, credit Football-Data.co.uk, and read
their disclaimer (https://www.football-data.co.uk/disclaimer.php) before using the data.

Download (5 small CSVs, about 1 MB in total, into `data/`):

```bash
python3 fetch_data.py    # or: ./download_data.sh
python3 features.py      # rebuilds out/matches_2526.json from data/
```

The files are updated by the source over time; results here were computed from the
versions downloaded on 2026-10-26. If the source later corrects a match, a rerun can differ slightly.
