#!/usr/bin/env python3
import argparse
import re

import requests


def download_file_from_google_drive(file_id: str, destination: str) -> None:
    """
    Download a file from Google Drive given its file ID.
    Handles confirmation token for large files.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # Initial request to get the download token (if needed)
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print(f"Downloaded file from Google Drive to '{destination}'.")


def get_confirm_token(response: requests.Response) -> str | None:
    """
    Check cookies for a confirmation token (used for large files).
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response: requests.Response, destination: str) -> None:
    """
    Save the content of the response in chunks.
    """
    CHUNK_SIZE = 32768  # 32KB
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive chunks.
                f.write(chunk)


def download_file_from_dropbox(url: str, destination: str) -> None:
    """
    Download a file from Dropbox.
    Adjusts the URL to force a direct download.
    """
    # Ensure the URL forces a direct download by using 'dl=1'
    if "dl=0" in url:
        url = url.replace("dl=0", "dl=1")
    elif "dl=1" not in url:
        if "?" in url:
            url += "&dl=1"
        else:
            url += "?dl=1"

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded file from Dropbox to '{destination}'.")
    else:
        print(f"Error downloading file from Dropbox: HTTP {response.status_code}")


def extract_google_drive_file_id(url: str) -> str:
    """
    Extract the file ID from a Google Drive URL.
    For example, given:
      https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    it returns 'FILE_ID'.
    """
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    # If no match, assume the URL is just the file id.
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Download file from a Google Drive or Dropbox URL")
    parser.add_argument("url", help="Google Drive or Dropbox URL")
    parser.add_argument("destination", help="Destination file path")
    args = parser.parse_args()

    url = args.url
    destination = args.destination

    if "drive.google.com" in url:
        file_id = extract_google_drive_file_id(url)
        download_file_from_google_drive(file_id, destination)
    elif "dropbox.com" in url:
        download_file_from_dropbox(url, destination)
    else:
        print("Error: URL must be either a Google Drive or Dropbox URL.")


if __name__ == "__main__":
    main()
