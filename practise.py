import csv, os
csv_path = "outputs/proposals/proposals.csv"
frames_dir = "outputs/frames"
with open(csv_path) as f:
    rows = list(csv.DictReader(f))
filenames = set([r['filename'] for r in rows])
for fn in filenames:
    assert os.path.exists(os.path.join(frames_dir, fn)), f"Missing frame: {fn}"
print("Basic CSV -> frames mapping OK; proposals count:", len(rows))