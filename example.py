#!/usr/bin/env python3
"""
Minimal reproduction: fit the baseline GARCH(1,1) on the DT50 low-carbon index
and print the parameter table with standard errors and in/out-of-sample MSE.

    pip install -r requirements.txt
    python example.py

Runs in well under a minute. The remaining five specifications live in
`Code/Forecasting Model Code/Code for Six GARCH Family Models/` and follow the
same call signature; the MIDAS variants additionally take an exogenous
indicator series and two MIDAS window lengths.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")          # no display needed
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "Code", "Forecasting Model Code",
                    "Code for Six GARCH Family Models")
DATA = os.path.join(CODE, "DT50低频.csv")

IN_SAMPLE = 1883               # ~70% of 2691 observations, as in the paper


def load_model(filename, symbol):
    """Import a fitting function from one of the model scripts.

    The scripts were written to be run directly and do their plotting in a
    module-level `__main__` block, so we execute only the part above it.
    """
    path = os.path.join(CODE, filename)
    with open(path, encoding="utf-8") as fh:
        source = fh.read().split("if __name__")[0]
    namespace = {"__file__": path}
    exec(compile(source, path, "exec"), namespace)
    return namespace[symbol]


def main():
    if not os.path.exists(DATA):
        sys.exit(f"Data file not found: {DATA}")

    data = pd.read_csv(DATA, encoding="gbk")
    rt = data["rt"].values

    print(f"DT50 low-carbon index — {len(rt)} daily observations, "
          f"{IN_SAMPLE} in sample / {len(rt) - IN_SAMPLE} out of sample\n")

    garch = load_model("GARCH.py", "garch")
    params, variance = garch(rt, IN_SAMPLE)

    print("\nPersistence check: alpha + beta =",
          round(float(params[2] + params[3]), 4),
          "\n(close to 1 is the high-persistence result reported in the paper)")


if __name__ == "__main__":
    main()
