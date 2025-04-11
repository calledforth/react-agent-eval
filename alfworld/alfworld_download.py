import os
import zipfile
import requests
from tqdm import tqdm


def download_alfworld_dataset():
    """
    Downloads the ALFWorld dataset from the specified URL and extracts it.
    """
    # URL for the ALFWorld dataset
    url = "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip"

    # Create a datasets directory if it doesn't exist
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    alfworld_dir = os.path.join(dataset_dir, "alfworld")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(alfworld_dir):
        os.makedirs(alfworld_dir)

    # Path to save the downloaded zip file
    zip_path = os.path.join(alfworld_dir, "alfworld_data.zip")

    # Download the file
    print(f"Downloading ALFWorld dataset from {url}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    # Download with progress bar
    with open(zip_path, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Extract the zip file
    print(f"Extracting dataset to {alfworld_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(alfworld_dir)

    # Remove the zip file after extraction
    os.remove(zip_path)

    print(f"ALFWorld dataset downloaded and extracted to {alfworld_dir}")
    print("Dataset structure:")
    for root, dirs, files in os.walk(alfworld_dir):
        level = root.replace(alfworld_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files[
            :5
        ]:  # Show only first 5 files in each directory to avoid clutter
            print(f"{sub_indent}{f}")
        if len(files) > 5:
            print(f"{sub_indent}... ({len(files) - 5} more files)")


if __name__ == "__main__":
    download_alfworld_dataset()
