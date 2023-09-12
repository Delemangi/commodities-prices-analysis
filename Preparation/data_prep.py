import argparse
import glob
import os

import pandas as pd

### Data preparation

# Steps
# 1. Load data
# 2. Rename important columns
# 3. Remove all unnecessary columns
# 4. Change Date column to ISO date format
# 5. Drop rows with missing values in column Title
# 6. Remove all duplicate rows
# 7. check for keywords in Title column
# 8. Save columns Title, Date, Link
# 9. Drop columns Link, Date
# 10. Remove all duplicate rows
# 11. Save column Title

# various formats
timestamp_format = "%Y-%m-%d %H:%M:%S"
date_format = "%Y-%m-%d"

# the columns we want in all prepped files
columns = ["Title", "Date", "Link"]


def gdelt_data_prep(file_path: str, output_path: str, keywords_path: str | None) -> None:
    """GDELT data preparation.

    Args:
        file_path (str): the input file path
        output_path (str): the output folder path
        keywords_path (str | None): the path to the filter keywords
    """

    # load data
    df = pd.read_csv(file_path)

    # remove all unnecessary columns
    df = df[columns]

    # change Date column to ISO date format
    df["Date"] = pd.to_datetime(df["Date"], format=timestamp_format).dt.strftime(date_format)

    # drop rows with missing values in column Title
    df.dropna(subset=["Title"], inplace=True)

    # data source agnostic steps
    data_prep(df, file_path, output_path, keywords_path)


def gnews_data_prep(file_path: str, output_path: str, keywords_path: str | None) -> None:
    """Google News data preparation.

    Args:
        file_path (str): the input file path
        output_path (str): the output folder path
        keywords_path (str | None): the path to the filter keywords
    """

    # load data
    df = pd.read_csv(file_path)

    # remove all unnecessary columns
    df = df[columns]

    data_prep(df, file_path, output_path, keywords_path)


def reddit_data_prep(file_path: str, output_path: str, keywords_path: str | None) -> None:
    """Reddit data preparation.

    Args:
        file_path (str): the input file path
        output_path (str): the output folder path
        keywords_path (str | None): the path to the filter keywords
    """

    # load data
    df = pd.read_csv(file_path)

    # rename important columns
    df.rename(columns={"title": "Title", "created_utc": "Date", "url": "Link"}, inplace=True)

    # remove all unnecessary columns
    df = df[columns]

    # convert unix timestamp to date
    df["Date"] = pd.to_datetime(df["Date"], unit="s").dt.strftime(date_format)

    data_prep(df, file_path, output_path, keywords_path)


def twitter_data_prep(file_path: str, output_path: str, keywords_path: str | None) -> None:
    """Twitter data preparation.

    Args:
        file_path (str): the input file path
        output_path (str): the output folder path
        keywords_path (str | None): the path to the filter keywords
    """

    # load data
    df = pd.read_csv(file_path)

    # rename important columns
    df.rename(columns={"tweet": "Title", "date": "Date", "link": "Link"}, inplace=True)

    # remove all unnecessary columns
    df = df[columns]

    # drop rows with missing values in important columns
    df.dropna(subset=["Title"], inplace=True)

    data_prep(df, file_path, output_path, keywords_path)


def data_prep(df: pd.DataFrame, file_path: str, output_path: str, keywords_path: str | None) -> None:
    """Data preparation.

    Args:
        df (pd.DataFrame): DataFrame with columns Title, Date, Link
        file_path (str): the input file path
        output_path (str): the output folder path
        keywords_path (str | None): the path to the filter keywords
    """

    # remove all duplicate rows
    df.drop_duplicates(inplace=True)

    # check for filter keywords in Title column
    if keywords_path is not None:
        try:
            with open(get_filter_keywords_file(os.path.basename(file_path), keywords_path), "r") as f:
                keywords = [k.strip() for k in f.readlines()]

            print(f"Found filter keywords for {os.path.basename(file_path)}")
            print(f"Length before filtering: {len(df)}")
            df = df[df["Title"].str.contains("|".join(keywords), case=False)].copy()
            print(f"Length after filtering: {len(df)}")
        except FileNotFoundError:
            print(f"No filter keywords for {os.path.basename(file_path)}")

    # check for missing values
    print(df.isna().sum())

    # save columns Title, Date, Link
    df.to_csv(os.path.join(output_path, f"{os.path.basename(file_path).split('.csv')[0]}_full.csv"), index=False)

    # drop columns Link, Date
    df.drop(columns=["Link", "Date"], inplace=True)

    # remove all duplicate rows
    df.drop_duplicates(inplace=True)

    # check for missing values
    print(df.isna().sum())

    # save column Title
    df.to_csv(os.path.join(output_path, f"{os.path.basename(file_path).split('.csv')[0]}_titles.csv"), index=False)


# the keys should be the first part of the files, which is the data source
data_prep_functions = {
    "gdelt": gdelt_data_prep,
    "googlenews": gnews_data_prep,
    "reddit": reddit_data_prep,
    "twitter": twitter_data_prep,
}


### Main


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: passed command line arguments
    """

    parser = argparse.ArgumentParser(description="Data Preparation")

    parser.add_argument("-i", "--input", type=str, default="input", help="Input folder path")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output folder path")
    parser.add_argument("-f", "--filter", type=str, default=None, help="Filter keywords folder path")

    return parser.parse_args()


def get_data_source(file: str) -> str:
    """Get the data source from the file name.

    Args:
        file (str): the file name

    Returns:
        str: the data source
    """

    return file.split("_")[0]


def get_keywords(file: str) -> str:
    """Get the keyword from the file name.

    Args:
        file (str): the file name

    Returns:
        str: the keyword
    """

    return file.split("_")[1]


def get_filter_keywords_file(file: str, folder: str) -> str:
    """Get the filter keywords file name.

    Args:
        file (str): the file name
        folder (str): the folder path

    Returns:
        str: the filter keywords file name
    """

    return os.path.join(folder, f"{get_keywords(file)}.txt")


def main() -> None:
    """Entry point.

    Raises:
        FileNotFoundError: if the input path or filter keywords path does not exist
        NotADirectoryError: if the input path is not a directory
    """

    args = parse_args()

    input_path: str = args.input
    output_path: str = args.output
    filter_path: str | None = args.filter

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path {input_path} does not exist")

    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Path {input_path} is not a directory")

    if filter_path is not None and not os.path.isdir(filter_path):
        raise FileNotFoundError(f"Path {filter_path} does not exist")

    files = glob.glob(os.path.join(input_path, "*.csv"))

    os.makedirs(output_path, exist_ok=True)

    for file in files:
        try:
            data_prep_functions[get_data_source(os.path.basename(file)).lower()](file, output_path, filter_path)
            print(f"Saved file {os.path.basename(file)}")
        except KeyError:
            print(f"Data source {get_data_source(file)} for file {file} not supported")

        print()


if __name__ == "__main__":
    main()
