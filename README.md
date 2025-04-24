# Wireless Security Project

## Description

WirelessSecProject is a comprehensive pipeline for wireless network security analysis utilizing the CIC-IDS2017 dataset. It provides modules to download and preprocess the dataset, train a machine learning model for intrusion detection, and deploy a REST API with Flask for real-time inference. A network sniffer captures live traffic and forwards it to the API for on-the-fly threat detection.

## Features

- Automatic download and extraction of the CIC-IDS2017 dataset
- Data cleaning, feature extraction, and class balancing
- Training of a deep learning model (LSTM/Dense) for intrusion detection
- Flask-based REST API for serving predictions
- Real-time network traffic sniffer that sends packet data to the API

## Requirements

- Python 3.7 or later
- Internet connection for dataset download
- Project dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd wirelessSecProject
   ```
2. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Configuration

- **constants.py**: Defines constants such as `DATASET_URL`, `DATASET_PATH`, `STORED_MODEL_PATH`, and `RANDOM_SEED`.
- **log.py**: Configures the global logger with timestamped entries and severity levels.

## Usage
1. **Dowload the data, preprocess the dataset and train the model**
   ```bash
   python model.py
   ```
2. **Run the Flask API**
   ```bash
   python flask_backend.py
   ```
   The API will be available at `http://127.0.0.1:5000`.
3. **Start the network sniffer (in the venv directory)**
   ```bash
   sudo /venv/bin/python net_scan.py
   ```
   The sniffer captures live traffic and sends it to the Flask endpoint for inference.

## Project Structure

```
├── constants.py                 # Project constants and paths
├── dataset_download.py          # Download and extract CIC-IDS2017 dataset
├── dataset_preprocessing.py     # Data cleaning, feature extraction, and balancing
├── model.py                     # Model training and evaluation script
├── flask_backend.py             # Flask REST API for real-time predictions
├── log.py                       # Logging configuration
├── net_scan.py                  # Network sniffer for live traffic forwarding
└── requirements.txt             # Python dependencies
```

## Logging

All modules use Python's built-in `logging` library to generate timestamped log entries with severity levels. Logs are printed to the console by default.

## Contact

Daniel Alamillo Martínez – danialamillo20@gmail.com

