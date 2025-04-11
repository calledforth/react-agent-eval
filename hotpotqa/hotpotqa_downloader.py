import requests
import json
import os


def download_hotpotqa_data(url, output_file="hotpot_dev_fullwiki_v1.json"):
    """
    Download JSON data from the provided URL and save it to a file.

    Args:
        url (str): URL to download the JSON data from
        output_file (str): Name of the file to save the JSON data to
    """
    print(f"Downloading data from {url}...")

    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        # Get the total file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Save the response content to a file
        with open(output_file, "wb") as f:
            if total_size > 0:
                print(f"Total file size: {total_size / (1024 * 1024):.2f} MB")

                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress
                        progress = downloaded / total_size * 100
                        print(
                            f"\rDownloading: {progress:.2f}% ({downloaded / (1024 * 1024):.2f} MB)",
                            end="",
                        )
            else:
                print("Downloading file (size unknown)...")
                f.write(response.content)

        print(f"\nDownload complete. File saved to {os.path.abspath(output_file)}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

    return True


if __name__ == "__main__":
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
    download_hotpotqa_data(url)
