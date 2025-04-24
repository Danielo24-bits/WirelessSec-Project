import os
import requests
import zipfile
import logging
from tqdm import tqdm
from constants import DATASET_URL, DATASET_PATH
from log import _logger

def extract_zip(zip_path, remove = False):
    """ Extracts a zip file to the same directory as the zip file."""
    _logger.info(f"Extracting zip file: {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            output_path = os.path.dirname(zip_path)
            zip_ref.extractall(output_path)

        _logger.info(f"Extraction complete. Files extracted to: {output_path}")
        if remove:
            os.remove(zip_path)
            _logger.info(f"Removed zip file: {zip_path}")

    except Exception as e:
        _logger.error(f"Error extracting zip file: {zip_path}: {e}")


def download_dataset(zip_name = 'cicids2017.zip', extract = True):
    """ Downloads the CIC-IDS2017 dataset"""
    ds_url = DATASET_URL
    ds_path = DATASET_PATH
    ds_zip_path = os.path.join(ds_path, zip_name)

    # Try to create directory if zip file doesn't exist
    if not os.path.exists(ds_zip_path):
        os.makedirs(ds_path, exist_ok = True)
    
        # Retrieve and store the dataset (monitoring progress with tqdm)
        response = requests.get(ds_url, stream = True)
        if response.status_code != 200:
            _logger.error(f"Failed to download dataset '{zip_name}' from '{ds_url}'. Status code: {response.status_code}")

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        _logger.info(f"Downloading dataset '{zip_name}' from '{ds_url}'...")
        with tqdm(total = total_size, unit = 'B', unit_scale = True, unit_divisor = 1024) as pbar:
            with open(ds_zip_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)

        # Check if the download was complete
        if total_size != 0 and pbar.n != total_size:
            _logger.error("ERROR, something went wrong during the download.")
        else:
            _logger.info(f"Download complete: {ds_zip_path}")

    else:
        _logger.warning(f"Dataset {ds_zip_path} already exists. Skipping download.")

        # Extract the dataset if [extract] option is True
        if extract:
            extract_zip(ds_zip_path, remove = False)
