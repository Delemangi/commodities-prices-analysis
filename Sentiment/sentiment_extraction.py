import argparse
import glob
import os
from datetime import datetime

import pandas as pd
from transformers import Pipeline, pipeline


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from given path.

    Args:
        path (str): .csv file path

    Returns:
        pd.DataFrame: DataFrame from the .csv file
    """

    return pd.read_csv(path)


def extract_sentiment(df: pd.DataFrame, classifier: Pipeline) -> list[float]:
    """
    Extract sentiments.

    Args:
        df (pd.DataFrame): the DataFrame to extract sentiments from
        classifier (Pipeline): the classifier to use

    Returns:
        list[float]: list of sentiment scores
    """

    sentiments = []
    titles = df["Title"].astype(str).tolist()
    sentiments = classifier(titles)
    scores = []

    for s in sentiments:
        if s["label"] == "positive":
            scores.append(s["score"])
        else:
            scores.append(1 - s["score"])

    return scores


def construct_df(titles_df: pd.DataFrame, full_df: pd.DataFrame, classifier: Pipeline) -> pd.DataFrame:
    """
    Construct a DataFrame.

    Args:
        titles_df (pd.DataFrame): DataFrame with just titles
        df_2 (pd.DataFrame): DataFrame with all data
        classifier (Pipeline): the classifier to use

    Returns:
        pd.DataFrame: DataFrame with all data and sentiment scores
    """

    scores = extract_sentiment(titles_df, classifier)
    titles_df.insert(1, "Sentiment", scores, True)
    df = pd.merge(titles_df, full_df, on="Title", how="outer")
    df.dropna(inplace=True)
    df.drop_duplicates(subset="Title", keep="first", inplace=True)

    return df


def save_df(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to .csv format.

    Args:
        df (pd.DataFrame): DataFrame to save
        path (str): .csv file path
    """

    df.to_csv(path, index=False)


def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily average sentiment score.

    Args:
        df (pd.DataFrame): DataFrame to calculate daily average sentiment score from

    Returns:
        pd.DataFrame: DataFrame with daily average sentiment score
    """

    df.drop(["Title", "Link"], axis=1, inplace=True, errors="ignore")
    df = df.groupby(["Date"]).mean().reset_index()
    df.sort_values(by=["Date"], inplace=True)

    return df


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
        raise NotADirectoryError("Input folder must exist")

    os.makedirs(output_path, exist_ok=True)

    classifier = pipeline("sentiment-analysis", model="mrm8488/deberta-v3-small-finetuned-sst2", device=0)
    files = set([i.replace("_titles", "").replace("_full", "") for i in glob.glob(os.path.join(input_path, "*.csv"))])

    for file in files:
        datetime_start = datetime.now()
        print("File:", os.path.basename(file))
        print("Started at:", datetime_start.strftime("%H:%M:%S"))

        titles_df = load_dataset(file.replace(".csv", "_titles.csv"))
        full_df = load_dataset(file.replace(".csv", "_full.csv"))

        df = construct_df(titles_df, full_df, classifier)
        save_df(df, os.path.join(output_path, "Sentiment_" + os.path.basename(file).replace(".csv", "_all.csv")))

        daily_df = daily_average(df)
        save_df(daily_df, os.path.join(output_path, "Sentiment_" + os.path.basename(file)))

        datetime_end = datetime.now()
        print("Ended at:", datetime_end.strftime("%H:%M:%S"))
        print("Duration:", datetime_end - datetime_start)
        print()


if __name__ == "__main__":
    main()
