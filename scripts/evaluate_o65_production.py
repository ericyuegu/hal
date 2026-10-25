"""Evaluate the explicitly selected final O65 checkpoint once."""

import tyro

from scripts.o65_production import EvaluationArgs
from scripts.o65_production import evaluate

if __name__ == "__main__":
    evaluate(tyro.cli(EvaluationArgs))
