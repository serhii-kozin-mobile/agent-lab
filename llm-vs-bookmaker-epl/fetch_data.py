"""Download Premier League results + odds from football-data.co.uk (free, public)."""
import pathlib, urllib.request
SEASONS = ["2122", "2223", "2324", "2425", "2526"]
out = pathlib.Path(__file__).parent / "data"; out.mkdir(exist_ok=True)
for s in SEASONS:
    url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    (out / f"E0_{s}.csv").write_bytes(urllib.request.urlopen(req).read())
    print("ok", url)
