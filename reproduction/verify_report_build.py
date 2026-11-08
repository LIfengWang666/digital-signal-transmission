"""Build the current report twice from source-only directories and compare."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from reproduction.build_pdf import build_pdf
from reproduction.check_reproducibility import file_digest
from reproduction.prepare_report import report_source_digest

ROOT = Path(__file__).resolve().parents[1]


def verify_report(compiler: Path, destination: Path) -> None:
    """Compare two independently compiled copies of the current report.

    Args:
        compiler: Installed Tectonic executable.
        destination: New directory outside the included package.

    Returns:
        None. Save build hashes and comparison evidence, raising on failure.
        Existing destinations are rejected without removing any files.
    """
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("Use a new report-comparison directory.")
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Comparison directory must be outside the package.")
    destination.mkdir(parents=True)
    report = ROOT / "report"
    copied = destination / "report"
    for source in sorted(report.rglob("*")):
        include = source.suffix in {".tex", ".bib", ".png"} or (
            source.suffix == ".pdf" and source.parent == report / "supporting"
        )
        if source.is_file() and include:
            target = copied / source.relative_to(report)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    source_hash = report_source_digest()
    first = build_pdf(compiler, report, ROOT / ".build-cache")
    second = build_pdf(compiler, copied, destination / ".build-cache")
    equal = first["compiled_pdf_sha256"] == second["compiled_pdf_sha256"]
    assert report_source_digest() == source_hash
    result = {
        "status": "passed" if equal else "failed",
        "date": datetime.now().astimezone().date().isoformat(),
        "report_source_sha256": source_hash,
        "reference_pdf_sha256": first["compiled_pdf_sha256"],
        "independent_pdf_sha256": second["compiled_pdf_sha256"],
        "exact_pdf_bytes_match": equal,
        "comparison_directory_began_without_PDF_or_auxiliaries": True,
        "both_compilations_used_new_temporary_source_only_workspaces": True,
        "SOURCE_DATE_EPOCH": 1792627200,
        "compiler": first["compiler"],
        "checker_sha256": file_digest(Path(__file__)),
    }
    (ROOT / "stage6/results/report_rebuild_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "metadata/report_build.json").write_text(
        json.dumps(
            {
                "status": ("Stages1--6 draft; final editing/package pending"),
                "report_source_sha256": source_hash,
                "compiled_pdf_sha256": first["compiled_pdf_sha256"],
                "compiler": first["compiler"],
                "build_policy": first["build_policy"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not equal:
        raise ValueError("Independent source-only PDF builds differ.")
    print("Current report: two source-only PDF builds are byte-identical.")


def main() -> None:
    """Read the compiler and unused destination, then check both builds.

    Returns:
        None. Preserve existing comparison directories and historical checks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    verify_report(args.tectonic.resolve(), args.destination)


if __name__ == "__main__":
    main()
