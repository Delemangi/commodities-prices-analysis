import argparse
import glob
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: passed command line arguments
    """

    parser = argparse.ArgumentParser(description="Data Preparation")

    parser.add_argument("-i", "--input", type=str, default="input", help="Input folder path")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output folder path")

    return parser.parse_args()


def main() -> None:
    """
    Entry point.

    Raises:
        NotADirectoryError: if the input or output folders don't exist
    """

    args = parse_args()

    input_path: str = args.input
    output_path: str = args.output

    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Input folder '{input_path}' does not exist")

    files = [file for file in glob.glob(os.path.join(input_path, "*.csv")) if "_all" not in file]

    names = [os.path.basename(file) for file in files]
    correlations = []

    start = "2019-03-01"
    end = "2023-03-01"
    date_range = pd.date_range(start, end, freq="D")

    for i in range(0, len(files) + 1):
        for j in range(i + 1, len(files)):
            file_1 = files[i]
            file_2 = files[j]

            x = pd.read_csv(file_1)
            y = pd.read_csv(file_2)

            x.set_index("Date", inplace=True)
            y.set_index("Date", inplace=True)

            x.index = pd.to_datetime(x.index)
            y.index = pd.to_datetime(y.index)

            x = x.reindex(date_range)
            y = y.reindex(date_range)

            x = x.rolling(window=3, min_periods=1).mean()
            y = y.rolling(window=3, min_periods=1).mean()

            x = x.bfill()
            y = y.bfill()

            x = x.ffill()
            y = y.ffill()

            if x.isnull().values.any():
                print(f"Missing values found in {os.path.basename(file_1)}")
                print(x[x.isnull().any(axis=1)])

            if x.size != y.size:
                print("Unequal sizes")
                print(x.size, y.size)

            pearson = x.iloc[:, 0].corr(y.iloc[:, 0], method="pearson")
            correlations.append([names[i].split("_")[2], names[j].split("_")[2], pearson])

    dataset = pd.DataFrame(correlations, columns=["A", "B", "Correlation"])
    dataset = dataset.sort_values(by="Correlation", key=abs, ascending=False)

    os.makedirs("output", exist_ok=True)
    dataset.to_csv(os.path.join(output_path, "output.csv"), index=False)


if __name__ == "__main__":
    main()
