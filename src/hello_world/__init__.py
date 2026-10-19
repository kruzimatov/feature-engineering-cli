import sys

import pandas as pd

from .features import one_hot, ordinal_encode, standard_scale

EDUCATION_ORDER = [
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: hello-world <csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df = ordinal_encode(df, "level of education", EDUCATION_ORDER)
    df = one_hot(df, "division")
    df = standard_scale(df, ["salary", "sales"])
    print(df)
