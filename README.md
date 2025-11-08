# PySpark Mini Course — Workshop Materials

This repository contains a small, self-contained learning workshop for PySpark. It's built for interactive exploration in Jupyter notebooks and for small, low-risk script additions to support lessons.

## Repository layout (short)
- `research/` — primary notebooks and lesson materials (main: `research/IntroductionToPySpark.ipynb`).
- `data/` — small sample datasets used by the notebooks (examples: `data/boston.csv`, `data/titanic.csv`).
- `pyspark_ml_project/` and `src/` — example code and an OOP-style PySpark project (see `src/main.py`).
- `requirements.txt` — Python dependencies for local development.
- `.github/copilot-instructions.md` — repo-specific guidance for AI agents and contributors.

## Quick start (macOS, zsh)

Minimum steps to get the notebooks and example code running locally.

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source venv/bin/activate
```

2) Install Python dependencies

```bash
pip install -r requirements.txt
```

Note: `pyspark` can be installed via `pip` (recommended for simple local runs) or you can point to a Spark distribution via `SPARK_HOME` and use `spark-submit`. If you install Spark separately, ensure `JAVA_HOME` and `SPARK_HOME` are set in your shell.

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
python -m pyspark_ml_project.main
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
# Option A — direct run using the venv python (recommended)
.venv/bin/python pyspark_ml_project/main.py

# Option B — module mode if PYTHONPATH is configured for the package
python -m pyspark_ml_project.main

# 4) open Jupyter Lab to explore notebooks
jupyter lab
```

If you use a Spark distribution instead of `pip`-installed `pyspark`, export these env vars first (example):

```bash
export JAVA_HOME="/Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home"
export SPARK_HOME="/usr/local/Cellar/apache-spark/3.4.0/libexec"
export PATH="$SPARK_HOME/bin:$PATH"
```

## Small additions you can ask me to make

- `scripts/preview_data.py`: CLI helper that loads `data/boston.csv`, prints schema, and shows the first N rows.
- `CONTRIBUTING.md`: short guidance for editing notebooks and committing pedagogical changes.
- Replace any absolute data paths in `research/IntroductionToPySpark.ipynb` with relative `data/` paths and ensure the SparkSession uses `master("local[*]")`.

### `scripts/preview_data.py` usage

After I add the script (created at `scripts/preview_data.py`), run it from the repository root like this:

```bash
# show first 5 rows (default)
python scripts/preview_data.py data/boston.csv

# show first 10 rows
python scripts/preview_data.py data/boston.csv --num 10
```

Or run with your Spark distribution via `spark-submit` (if you need a full Spark install):

```bash
# when SPARK_HOME is set
$SPARK_HOME/bin/spark-submit --master local[*] scripts/preview_data.py data/boston.csv --num 10
```

## Troubleshooting

- If Spark fails to start: check `JAVA_HOME`, `SPARK_HOME`, and that the Python environment can import `pyspark`.
- If a notebook references a missing file, search for absolute paths and replace with `data/` relative paths.

## Next steps

I can implement any of the small additions above. Tell me which one you'd like first and I'll:

1. add the script and a short usage section to this README, or
2. update the main tutorial notebook to use relative paths and a local Spark master.

If you'd like, I can also run a quick verification that `python -m src.main` runs in this environment and report back.



