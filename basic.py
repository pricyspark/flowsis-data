import sys
import csv
from pathlib import Path

tool_ids = sorted(x.name for x in Path("images").iterdir() if x.is_dir())

sort_column_index = 0

with open("image_manifest.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

with open("previous_image_manifest.csv", mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

num_rows = len(rows)
num_cols = len(rows[0])

prev_tool_ids = set([row[0] for row in rows])

for id in tool_ids:
    if id in prev_tool_ids:
        print(f"ID {id} is already in manifest. Skipped.")
        continue
    
    subdir = Path(f"images/{id}")
    if not subdir.is_dir():
        print(f"images/{id} is not a directory. Skipped.")
    
    for img in subdir.iterdir():
        if not img.is_file():
            print(f"images/{id}/{img} is not a file. Skipped.")
            
        row = [''] * num_cols
        row[0] = id
        row[2] = f"images/{id}/{img.name}"
        rows.append(row)
    
rows = sorted(rows, key=lambda row: row[sort_column_index])
with open("image_manifest.csv", mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
