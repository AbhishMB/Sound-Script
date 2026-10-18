import os
import requests
import zipfile
from io import BytesIO

# URL for the Kaggle dataset zip (public dataset, no auth needed)
url = "https://www.kaggle.com/api/v1/datasets/download/imsparsh/musicnet-dataset?dataset_version_number=1"

print("Downloading only the 'musicnet' folder from the dataset...")

# Download the zip file
response = requests.get(url, stream=True)
response.raise_for_status()

# Load the zip into memory (since it's large, but we extract selectively)
zip_content = BytesIO(response.content)

# Open the zip and extract only the 'musicnet' folder
with zipfile.ZipFile(zip_content) as zf:
    # List files in the zip
    file_list = zf.namelist()
    # Filter for files in 'musicnet/' directory
    musicnet_files = [f for f in file_list if f.startswith("musicnet/")]

    if not musicnet_files:
        print("Error: 'musicnet' folder not found in the dataset.")
        exit(1)

    # Extract only the musicnet files
    for file in musicnet_files:
        zf.extract(file, ".")

print("Downloaded and extracted only the 'musicnet' folder to the current directory.")
