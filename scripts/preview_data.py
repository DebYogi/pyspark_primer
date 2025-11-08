#!/usr/bin/env python3
"""Small CLI to preview a CSV using PySpark.

Usage examples:
    python scripts/preview_data.py data/boston.csv --num 5
    $SPARK_HOME/bin/spark-submit --master local[*] scripts/preview_data.py data/boston.csv --num 10

The script reads the CSV (header, inferSchema), prints the schema, and shows the first N rows.
"""
import argparse
from pyspark.sql import SparkSession
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Preview CSV files using a local SparkSession")
    p.add_argument("path", help="Path to CSV file (relative to repo root or absolute)")
    p.add_argument("--num", "-n", type=int, default=5, help="Number of rows to show (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        spark = SparkSession.builder.appName("preview_data").master("local[*]").getOrCreate()
    except Exception as e:
        print("Failed to create SparkSession:", e, file=sys.stderr)
        sys.exit(2)

    try:
        df = spark.read.csv(args.path, header=True, inferSchema=True)
    except Exception as e:
        print(f"Failed to read CSV at {args.path}: {e}", file=sys.stderr)
        spark.stop()
        sys.exit(3)

    print("Schema:")
    df.printSchema()
    print(f"\nShowing first {args.num} rows:")
    df.show(args.num, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
