import argparse
import json
import os
import time
import urllib
from datetime import datetime
from typing import cast

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# creating a Session object makes sending the requests significantly faster
session = requests.Session()
retries = Retry(total=12, backoff_factor=4, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=1, pool_maxsize=1)
session.mount("https://", adapter)
session.mount("http://", adapter)

# various formats
gdelt_format = "%Y%m%d%H%M%S"
seen_date_format = "%Y%m%dT%H%M%SZ"
file_name_format = "%Y-%m-%d"
log_format = "%Y-%m-%d %H:%M:%S"

# GDELT does not let you search if the interval is less than 30 minutes
smallest_interval = 1800


def get_data_recursively(file_path: str, timestamp_from: datetime, timestamp_to: datetime, keywords: str, language: str, timeout: float) -> None:
    """Get data from GDELT.
    The intervals are split recursively until the number of articles is less than 250.
    This function will obtain more data, but is more prone to rate limiting.
    The smallest interval is 30 minutes.

    Args:
        file_path (str): name of the .csv file to be saved
        timestamp_from (datetime): beginning timestamp
        timestamp_to (datetime): ending timestamp
        keywords (str): keywords to search for
        language (str): language to search in
        timeout (float): timeout between requests
    """

    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={transform_keywords(keywords)}%20sourcelang:{language}&mode=ArtList&maxrecords=250&sort=DateAsc&format=json"
    start, end = cast(tuple[datetime, datetime], pd.date_range(timestamp_from, timestamp_to, periods=2).to_pydatetime().tolist())
    result = get_data_by_date(url, start, end, timeout)

    if len(result) == 250:
        print(f"[{timestamp_from.strftime(log_format)} - {timestamp_to.strftime(log_format)}] 250, halving")
        start, mid, end = cast(tuple[datetime, datetime, datetime], pd.date_range(timestamp_from, timestamp_to, periods=3).to_pydatetime().tolist())

        if (mid - start).total_seconds() >= smallest_interval and (end - mid).total_seconds() >= smallest_interval:
            get_data_recursively(file_path, start, mid, keywords, language, timeout)
            get_data_recursively(file_path, mid, end, keywords, language, timeout)
            return

    result.to_csv(file_path, encoding="utf-8", index=False, header=not os.path.exists(file_path), mode="a")
    print(f"[{timestamp_from.strftime(log_format)} - {timestamp_to.strftime(log_format)}] {result.shape[0]}, saving")


def get_data_iteratively(file_path: str, timestamp_from: datetime, timestamp_to: datetime, keywords: str, language: str, timeout: float, freq: str) -> None:
    """Get data from GDELT.
    The intervals are split by the given frequency.
    This function will obtain less data, but is less prone to rate limiting.

    Args:
        file_path (str): name of the .csv file to be saved
        timestamp_from (datetime): beginning timestamp
        timestamp_to (datetime): ending timestamp
        keywords (str): keywords to search for
        language (str): language to search in
        timeout (float): timeout between requests
        freq (str): frequency of the intervals
    """

    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={transform_keywords(keywords)}%20sourcelang:{language}&mode=ArtList&maxrecords=250&format=json"
    days = cast(list[datetime], pd.date_range(timestamp_from, timestamp_to, freq=freq).to_pydatetime().tolist())

    for day in days:
        start = datetime(day.year, day.month, day.day, 0, 0, 0)
        end = datetime(day.year, day.month, day.day, 23, 59, 59)

        result = get_data_by_date(url, start, end, timeout)
        result.to_csv(file_path, encoding="utf-8", index=False, header=not os.path.exists(file_path), mode="a")
        print(f"[{start.strftime(log_format)} - {end.strftime(log_format)}] {result.shape[0]}, saving")


def get_data_by_date(base_url: str, timestamp_from: datetime, timestamp_to: datetime, timeout: float) -> pd.DataFrame:
    """Get data from GDELT between two timestamps.

    Args:
        base_url (str): URL to send a request to
        timestamp_from (datetime): beginning timestamp
        timestamp_to (datetime): ending timestamp
        timeout (float): timeout between requests

    Returns:
        tuple[pd.DataFrame, datetime]: DataFrame with the data
    """

    request_url = f"{base_url}&startdatetime={timestamp_from.strftime(gdelt_format)}&enddatetime={timestamp_to.strftime(gdelt_format)}"
    articles = send_request(request_url, timeout)
    articles_df = pd.DataFrame(articles)
    data = []

    # this is all the available data, a different script should be used for taking only the relevant columns
    for item in articles_df.itertuples():
        url: str = item.articles["url"]
        url_mobile: str = item.articles["url_mobile"]
        title: str = item.articles["title"]
        seendate: str = item.articles["seendate"]
        socialimage: str = item.articles["socialimage"]
        domain: str = item.articles["domain"]
        language: str = item.articles["language"]
        sourcecountry: str = item.articles["sourcecountry"]

        data.append((url, url_mobile, title, datetime.strptime(seendate, seen_date_format).strftime(log_format), socialimage, domain, language, sourcecountry))

    return pd.DataFrame(data, columns=("Link", "Mobile Link", "Title", "Date", "Image", "Domain", "Language", "Country"))


def send_request(url: str, timeout: float) -> dict[str, str]:
    """Send a request to GDELT.

    Args:
        url (str): URL to send a request to
        timeout (float): timeout between requests

    Returns:
        dict[str, str]: parsed JSON from the response, or an empty dict if the response is not valid JSON
    """

    # sleep so we don't get rate limited as easily
    time.sleep(timeout)

    request = session.get(url)
    return parse_json(strip_invalid_characters(request.text))


def parse_json(data: str) -> dict[str, str]:
    """Parse JSON from GDELT, and fix it if necessary.

    Args:
        data (str): JSON to parse

    Returns:
        dict[str, str]: parsed JSON
    """

    try:
        return cast(dict[str, str], json.loads(data))
    except json.JSONDecodeError as e:
        return parse_json(data[: e.pos] + data[e.pos + 1 :])


def strip_invalid_characters(data: str) -> str:
    """Escape any backslashes, and remove any newline characters in the string.

    Args:
        data (str): the input JSON string

    Returns:
        str: the transformed JSON string
    """

    # more edge cases may be added with time
    # current edge cases:
    # 1. unescaped backslashes
    # 2. newlines are causing problems in strings
    # 3. some unicode characters are causing problems
    return data.replace("\\", "\\\\").replace("\n", "")


def transform_keywords(keywords: str) -> str:
    """Transform keywords to URL encoded GDELT format.

    Args:
        keywords (str): comma separated list of keywords to be transformed

    Returns:
        str: transformed keywords in GDELT format
    """

    if "," in keywords:
        sep = '" OR "'
        encoded_keywords = map(urllib.parse.quote, keywords.split(","))

        # multi keyword queries require double quotes around each keyword, separated by OR (with spaces)
        return f'("{sep.join(encoded_keywords)}")'
    else:
        encoded_keyword = urllib.parse.quote(keywords)

        # encoded keywords require double quotes, and also double quoted keywords need to be at least 5 characters long
        return f'"{encoded_keyword}"' if encoded_keyword != keywords else encoded_keyword


def transform_string_to_iso_datetime(date: str) -> datetime:
    """Transform a date or timestamp from ISO format to datetime.

    Args:
        date (str): date to be transformed

    Returns:
        datetime: transformed date in datetime format
    """

    return datetime.strptime(date, log_format) if " " in date else datetime.strptime(date, file_name_format)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: passed command line arguments
    """

    parser = argparse.ArgumentParser(description="Scrape GDELT")

    parser.add_argument("-n", "--name", type=str, help="Name of the .csv file to be saved")
    parser.add_argument("-k", "--keywords", type=str, required=True, help="Keywords, separated by commas (OR) or spaces (AND)")
    parser.add_argument("-l", "--language", type=str, default="eng", help="Language")
    parser.add_argument("-f", "--from", type=transform_string_to_iso_datetime, default=transform_string_to_iso_datetime("2019-03-01"), dest="from_", help=f"Beginning date or timestamp in ISO format ({log_format} or {file_name_format})")
    parser.add_argument("-t", "--to", type=transform_string_to_iso_datetime, default=transform_string_to_iso_datetime("2023-03-01"), help=f"Ending date or timestamp in ISO format ({log_format} or {file_name_format})")
    parser.add_argument("-to", "--timeout", type=float, default=0.5, help="Timeout in seconds")
    parser.add_argument("-s", "--strategy", type=str, default="recursive", help="Strategy to use when scraping (r for recursive, i for iterative)")
    parser.add_argument("-freq", "--frequency", type=str, default="1D", help="Frequency to use when scraping (1D, 12H, etc. See pandas docs for more info)")

    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()

    name: str = args.name
    keywords: str = args.keywords
    language: str = args.language
    date_from: datetime = args.from_
    date_to: datetime = args.to
    timeout: float = args.timeout
    strategy: str = args.strategy
    frequency: str = args.frequency

    if name is None:
        name = keywords

    if not os.path.exists("./results"):
        os.makedirs("./results", exist_ok=True)

    strategy = strategy.lower()

    if strategy == "r":
        strategy = "recursive"
    elif strategy == "i":
        strategy = "iterative"
    if strategy != "recursive" and strategy != "iterative":
        strategy = "recursive"

    # recommended file name
    file_path = os.path.join("results", f"GDELT_{name}_{date_from.strftime(file_name_format)}_{date_to.strftime(file_name_format)}.csv")

    print(f"Getting data for keywords {keywords} between {date_from.strftime(log_format)} and {date_to.strftime(log_format)} in language {language} with timeout {timeout} seconds and strategy {strategy}")

    if strategy == "iterative":
        get_data_iteratively(file_path, date_from, date_to, keywords, language, timeout, frequency)
    else:
        get_data_recursively(file_path, date_from, date_to, keywords, language, timeout)

    print(f"Done. Saved to {file_path}")


if __name__ == "__main__":
    main()
