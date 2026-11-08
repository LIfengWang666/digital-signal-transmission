"""Generate reproducibility report fragments from measured evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "stage6"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write an evidence ledger with explicit header fields.

    Args:
        path: Output CSV destination.
        rows: Nonempty homogeneous dictionaries of evidence fields.

    Returns:
        None. Preserve literal dates, versions and role descriptions.
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: str) -> str:
    """Escape literal record text for safe TeX table cells.

    Args:
        value: Plain text without LaTeX markup.

    Returns:
        Equivalent text with TeX-special punctuation escaped.
    """
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(c, c) for c in value)


def prepare_r61() -> None:
    """Format counts from this invocation's product verification.

    Returns:
        None. Generate the measured product-count macro fragment.
    """
    (ROOT / "report/generated").mkdir(parents=True, exist_ok=True)
    (FOLDER / "results").mkdir(parents=True, exist_ok=True)
    verified = json.loads(
        (FOLDER / "results/product_verification.json").read_text(
            encoding="utf-8"
        )
    )
    if verified["status"] != "passed":
        raise ValueError("The clean run did not verify all products.")
    kinds = verified["kinds"]
    macros = {
        "RsixProducts": str(verified["products_checked"]),
        "RsixCsv": str(kinds["measured/check CSV"]),
        "RsixPng": str(kinds["generated raster figure"]),
        "RsixSvg": str(kinds["generated vector figure"]),
        "RsixArrays": str(kinds["generated numerical array"]),
        "RsixFragments": str(kinds["generated LaTeX number/table"]),
    }
    (ROOT / "report/generated/r61_measurements.tex").write_text(
        "% Verified clean-data counts; final PDF gate is separate.\n"
        + "\n".join(r"\newcommand{\%s}{%s}" % item for item in macros.items())
        + "\n",
        encoding="utf-8",
    )
    print("R6.1 report counts prepared from current verified products.")


if __name__ == "__main__":
    prepare_r61()
