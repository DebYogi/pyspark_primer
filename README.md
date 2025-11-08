# PySpark Mini Course — Workshop Materials

[![Python Tests](https://github.com/DebYogi/pyspark_primer/actions/workflows/python-test.yml/badge.svg)](https://github.com/DebYogi/pyspark_primer/actions/workflows/python-test.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PySpark](https://img.shields.io/badge/pyspark-3.5.7-orange)](https://spark.apache.org/docs/latest/api/python/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a small, self-contained learning workshop for PySpark. It's built for interactive exploration in Jupyter notebooks and for small, low-risk script additions to support lessons.

## Repository layout
Main components:
- `research/` — Jupyter notebooks for interactive learning
  - `IntroductionToPySpark.ipynb` - main tutorial notebook
  - `sparkML_*.ipynb` - machine learning examples
  - `DataIO.ipynb`, `workingWithCols.ipynb` - data manipulation tutorials
  - Other supplementary notebooks for specific topics

- `data/` — Sample datasets in various formats
  - `boston.csv` - Boston housing dataset
  - `titanic.csv` - Titanic passenger data
  - Various partitioned and parquet versions for I/O examples

- `pyspark_ml_project/` — ML pipeline examples
  - `main.py` - Entry point showing PySpark ML pipeline orchestration
  - `keras_xgb_compare.py` - Comparison of Keras and XGBoost models
  - `src/` - Modular implementation (data loading, preprocessing, training)
    - `config.py` - Configuration and parameters
    - `data.py` - Data loading and preprocessing
    - `model.py` - ML model implementations
    - `evaluate.py` - Evaluation metrics

- `scripts/` — Utility scripts
  - `preview_data.py` - CLI tool for quick data exploration

- Project files
  - `requirements.txt` — Python dependencies
  - `.github/workflows/python-test.yml` - CI pipeline configuration
  - `LICENSE` - MIT license

## Quick start (macOS, zsh)

Minimum steps to get the notebooks and example code running locally.

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Note: use .venv, not venv
```

2) Install Python dependencies

```bash
pip install -r requirements.txt
```

Note: This installs all required packages including:
- `pyspark==3.5.7` - core PySpark functionality
- `jupyter` - for running the notebooks
- `numpy`, `pandas` - for data manipulation examples
- Other ML-related packages used in the examples

If you prefer using a separate Spark distribution instead of pip-installed PySpark, ensure `JAVA_HOME` and `SPARK_HOME` are set in your shell.

3) Start Jupyter Lab (open the notebooks in `research/`)

```bash
jupyter lab
```

4) Run the example project script (from repo root)

The example project entry point is `pyspark_ml_project/main.py` (it uses the package under `pyspark_ml_project/src`). To avoid ambiguity with a top-level `src` folder, run the script directly with the virtualenv Python executable:

```bash
# after creating & activating the venv (see Quick start above)
.venv/bin/python pyspark_ml_project/main.py
```

If you prefer module mode and your PYTHONPATH is configured to use the `pyspark_ml_project` package, you can run:

```bash
python -m main_keras_xgb.main
```

Avoid `python -m src.main` from the repository root — this may import the top-level `src` folder instead of the intended package and cause import errors.

Alternative: run scripts with Spark's `spark-submit` if you rely on a separate Spark distribution:

```bash
# when SPARK_HOME is set
$SPARK_HOME/bin/spark-submit --master local[*] --py-files pyspark_ml_project/src pyspark_ml_project/main.py
```

## Project conventions and tips

- Use relative `data/` paths inside notebooks and scripts. For example:

```py
# prefer
df = spark.read.csv("data/boston.csv", header=True, inferSchema=True)
```

- Create SparkSession with an explicit local master in workshop code so results are reproducible:

```py
from pyspark.sql import SparkSession
spark = SparkSession.builder \
	.appName("PySpark 101") \
	.master("local[*]") \
	.getOrCreate()
```

- Notebooks are didactic. Preserve explanatory markdown and comments; if you remove or change pedagogical content, include a short justification in the commit message.

## Try it — quick commands (zsh)

Copy/paste these commands in a macOS zsh prompt while at the repository root.

```bash
# 1) create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) run the example project
# Option A — PySpark ML Pipeline example
.venv/bin/python pyspark_ml_project/main.py

# Option B — Keras & XGBoost comparison
.venv/bin/python pyspark_ml_project/keras_xgb_compare.py

# 4) open Jupyter Lab to explore notebooks
jupyter lab
```

If you use a Spark distribution instead of `pip`-installed `pyspark`, export these env vars first (example):

```bash
export JAVA_HOME="/Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home"
export SPARK_HOME="/usr/local/Cellar/apache-spark/3.4.0/libexec"
export PATH="$SPARK_HOME/bin:$PATH"
```

## Additional Tools and Scripts

### Data Preview CLI (`scripts/preview_data.py`)

A quick way to inspect CSV files using PySpark:

```bash
# Activate your environment first
source .venv/bin/activate

# Show first 5 rows (default)
python scripts/preview_data.py data/boston.csv

# Show first 10 rows with schema
python scripts/preview_data.py data/boston.csv --num 10
```

Alternative: use `spark-submit` if you have a separate Spark installation:

```bash
$SPARK_HOME/bin/spark-submit --master local[*] scripts/preview_data.py data/boston.csv --num 10
```

## Troubleshooting

Common issues and solutions:

1. PySpark import errors
   - Ensure you've activated the virtualenv: `source .venv/bin/activate`
   - Verify PySpark is installed: `pip list | grep pyspark`
   - If using system Spark: check `JAVA_HOME` and `SPARK_HOME` are set

2. Notebook data loading
   - Use relative paths from repository root (e.g., `data/boston.csv`)
   - If you see absolute paths, replace them with relative ones
   - Run cells in order (SparkSession must be created first)

3. ML Pipeline issues
   - Ensure numpy/pandas are installed: `pip install numpy pandas`
   - Check training data exists in expected location
   - Use `.master("local[*]")` in SparkSession for local runs

## Contributing

This is a learning-focused repository. When contributing:
1. Keep notebooks educational and well-commented
2. Add docstrings to utility functions
3. Test changes with the example notebooks
4. Run the CI checks before submitting PRs

For detailed guidelines, see our [Contributing Guide](CONTRIBUTING.md).



