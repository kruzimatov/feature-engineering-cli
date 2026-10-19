# feature-engineering-cli

Feature engineering functions refactored from a Jupyter notebook into a tested Python package with a command-line entry point. Built for the HumbleBee AI roadmap (Software Engineering for AI).

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run hello-world data/office_sales.csv
```

The command reads the CSV, ordinal-encodes `level of education`, one-hot encodes `division`, standard-scales `salary` and `sales`, and prints the result.

## Test

```bash
uv run pytest tests/
```

## Functions

All functions live in `src/hello_world/features.py`. Each returns a new DataFrame and never changes the input.

| Function | What it does |
|---|---|
| `one_hot(df, column)` | One 0/1 column per category |
| `dummy(df, column)` | Like `one_hot`, but drops the first category |
| `ordinal_encode(df, column, order)` | Maps ordered categories to 0, 1, 2, ... and raises `ValueError` on a value not in `order` |
| `standard_scale(df, columns)` | Scales to mean 0, std 1. A constant column becomes 0 |
| `min_max_scale(df, columns)` | Scales to the range 0 to 1. A constant column becomes 0 |

## Project layout

```
src/hello_world/
  __init__.py     CLI entry point (main)
  features.py     feature engineering functions
tests/
  test_features.py
data/
  office_sales.csv
notebooks/
  Feature_Engineering_Exercises_done.ipynb   original notebook
```

## Data

`data/office_sales.csv` is the first 20 rows of `dm_office_sales.csv` (office supply sales by employee: division, education, training, experience, salary, sales).
