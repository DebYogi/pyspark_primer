"""Data ingestion module"""
import urllib.request
from pathlib import Path
from pyspark.sql import SparkSession

class DataLoader:
    def __init__(self, spark: SparkSession, url: str, local_path: Path):
        self.spark = spark
        self.url = url
        self.local_path = local_path

    def download_data(self):
        print(f"⬇️  Downloading data from {self.url} to {self.local_path}")
        urllib.request.urlretrieve(self.url, str(self.local_path))
        print("✅ Download complete.")

    def load_csv(self):
        print(f"📄 Loading CSV from {self.local_path}")
        df = self.spark.read.csv(str(self.local_path), header=True, inferSchema=True)
        df.show(5)
        return df
