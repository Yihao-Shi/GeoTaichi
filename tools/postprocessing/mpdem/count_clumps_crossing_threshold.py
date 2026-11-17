import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def process_npz_folder(
    folder_path: str | Path,
    output_excel: str | Path,
    x_threshold: float = 1.0,
):
    folder_path = Path(folder_path)
    output_excel = Path(output_excel)
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    file_pattern = re.compile(r"DEMClump(\d{6})\.npz")

    files = sorted(
        (path for path in folder_path.iterdir() if file_pattern.fullmatch(path.name)),
        key=lambda path: int(file_pattern.fullmatch(path.name).group(1)),
    )

    records = []

    for path in files:
        with np.load(path) as data:
            x_all = data["centerOfMass"][:, 0]
        count_gt = int((x_all > x_threshold).sum())
        records.append({
            "File Name": path.name,
            f"Count (x > {x_threshold})": count_gt,
        })

    pd.DataFrame.from_records(records).to_excel(
        output_excel, index=False, sheet_name="Count_Stats")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count DEM clump centers beyond an x threshold in recorder NPZ files."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing DEMClumpNNNNNN.npz files")
    parser.add_argument("output_excel", type=Path, help="Destination .xlsx workbook")
    parser.add_argument("--x-threshold", type=float, default=1.0)
    args = parser.parse_args()

    count = process_npz_folder(args.input_dir, args.output_excel, args.x_threshold)
    print(f"processed {count} files; wrote {args.output_excel}")


if __name__ == "__main__":
    main()
