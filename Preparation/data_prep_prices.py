import argparse
import glob
import os

import pandas as pd

# various formats
timestamp_format = "%Y-%m-%d %H:%M:%S"
date_format = "%Y-%m-%d"


def data_prep(file_path: str, output_path: str) -> None:
    """Data preparation.

    Args:
        file_path (str): the input file path
        output_path (str): the output folder path
    """

    df = pd.read_csv(file_path)

    # change Date column to ISO date format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.strftime(date_format)

    # drop column Vol.
    df.drop(columns=["Vol."], inplace=True)

    # change column Price to float
    if df["Price"].dtype == "object":
        df["Price"] = df["Price"].str.replace(",", "")
        df["Price"] = df["Price"].astype(float)

    # change column Open to float
    if df["Open"].dtype == "object":
        df["Open"] = df["Open"].str.replace(",", "")
        df["Open"] = df["Open"].astype(float)

    # change column High to float
    if df["High"].dtype == "object":
        df["High"] = df["High"].str.replace(",", "")
        df["High"] = df["High"].astype(float)

    # change column Low to float
    if df["Low"].dtype == "object":
        df["Low"] = df["Low"].str.replace(",", "")
        df["Low"] = df["Low"].astype(float)

    # remove all duplicate rows
    df.drop_duplicates(inplace=True)

    # save only columns Title, Date, Link
    df.to_csv(os.path.join(output_path, os.path.basename(file_path)), index=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: passed command line arguments
    """

    parser = argparse.ArgumentParser(description="GDELT Data Preparation")

    parser.add_argument("-i", "--input", type=str, default="input", help="Input folder path")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output folder path")

    return parser.parse_args()


def main() -> None:
    """Entry point.

    Raises:
        FileNotFoundError: if the input path does not exist
        NotADirectoryError: if the input path is not a directory
    """

    args = parse_args()

    input_path: str = args.input
    output_path: str = args.output

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path {input_path} does not exist")

    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Path {input_path} is not a directory")

    files = glob.glob(os.path.join(input_path, "*.csv"))

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for file in files:
        data_prep(file, output_path)


if __name__ == "__main__":
    main()
