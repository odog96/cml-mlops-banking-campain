import requests
import os

# URL for the bank-full.csv file (with separators as semicolons)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
zip_path = "data/bank.zip"
csv_path = "data/bank-full.csv"

# Create data directory
os.makedirs("data", exist_ok=True)

# Download the zip file
response = requests.get(url)
with open(zip_path, "wb") as f:
    f.write(response.content)

# Unzip to get bank-full.csv
from zipfile import ZipFile
with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("data")

# Verify
print(os.listdir("data"))  # Should include 'bank-full.csv'