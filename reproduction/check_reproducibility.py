"""Check the frozen R1--R5 products and write their reproducibility ledger."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# This module is imported before the numerical modules by run_all.py.
# Record chunks provide parallelism. The isolated R2.2 scan overrides this
# with its verified four-thread setting; other kernels use one thread.
os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
)


def file_digest(path: Path) -> str:
    """Hash a saved artifact without interpreting its numerical contents.

    Args:
        path: Existing source, input or generated product.

    Returns:
        Lowercase SHA-256 of the complete file bytes.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_versions() -> dict[str, str]:
    """Identify the numerical/rendering versions used for frozen records.

    Returns:
        Python, the permitted numerical libraries and rendering dependencies.
    """
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "scipy",
                "matplotlib",
                "Pillow",
                "contourpy",
                "cycler",
                "fonttools",
                "kiwisolver",
                "packaging",
                "pyparsing",
                "python-dateutil",
                "six",
            ]
        },
    }


def require_environment() -> None:
    """Require the validated versions for byte comparison of frozen results.

    Returns:
        None. Fail before experiment execution if the numerical stack differs.
        Exact replay requires this validated pin.
    """
    evidence = json.loads(
        (ROOT / "stage6/environment_evidence.json").read_text(encoding="utf-8")
    )
    expected = {
        "python": evidence["python"],
        **evidence["numeric_and_render_packages"],
    }
    actual = environment_versions()
    if actual != expected:
        raise RuntimeError(
            "Frozen product hashes require the validated environment: "
            f"{expected}; found {actual}. Use that environment rather "
            "than rewriting experimental freezes to suppress differences."
        )


def main() -> None:
    """Verify all frozen scientific products under explicit comparison rules.

    Returns:
        None. Save a product ledger and raise if any scientific target differs.
        Targets are never regenerated from a failed run.
    """
    from reproduction.scientific_fingerprint import scientific_digest

    require_environment()
    baseline_path = ROOT / "stage6/baseline_products.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows, changed = [], []
    for item in baseline["products"] + baseline["packing_products"]:
        path = ROOT / item["path"]
        actual = (
            scientific_digest(path, item["comparison"])
            if path.is_file()
            else "missing"
        )
        equal = actual == item["sha256"]
        if not equal:
            changed.append(item["path"])
        rows.append(
            {
                "path": item["path"],
                "kind": item["kind"],
                "comparison": item["comparison"],
                "sha256": actual,
                "matches_baseline": int(equal),
            }
        )
    if changed:
        raise ValueError(f"Scientific product mismatches: {changed}")
    destination = ROOT / "stage6/results"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "product_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "passed",
        "products_checked": len(baseline["products"]),
        "packing_products_checked": len(baseline["packing_products"]),
        "kinds": dict(Counter(r["kind"] for r in baseline["products"])),
        "baseline_sha256": file_digest(baseline_path),
        "numerical_environment": environment_versions(),
        "new_independent_observations": 0,
        "comparison_scope": "Exact CSV values, arrays, payloads and TeX; exact image pixels; SVG visible content with metadata/opaque IDs excluded",
    }
    (destination / "product_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Reproducibility: {len(rows)} scientific products match their fixed targets."
    )


if __name__ == "__main__":
    main()
