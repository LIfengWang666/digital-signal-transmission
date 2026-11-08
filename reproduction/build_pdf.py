"""Compile only report inputs in a clean deterministic TeX workspace."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_pdf(compiler: Path, report: Path, cache: Path) -> dict[str, str]:
    """Build the report without inheriting stale TeX auxiliary files.

    Args:
        compiler: Installed Tectonic0.17.0 executable.
        report: Source directory with main.tex, sections, images and cover.
        cache: Project-owned dependency/temporary-build directory.

    Returns:
        Compiler version and output digest. Copy PDF/logs/intermediates
        back to report; experiment data and report inputs are unchanged.
    """
    if not compiler.is_file() or not (report / "main.tex").is_file():
        raise FileNotFoundError("Compiler or report main source is missing.")
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("TECTONIC_CACHE_DIR", str(cache / "tectonic"))
    environment["SOURCE_DATE_EPOCH"] = "1792627200"
    with tempfile.TemporaryDirectory(prefix="latex_", dir=cache) as folder:
        build = Path(folder).resolve()
        if not build.is_relative_to(cache.resolve()):
            raise ValueError("Temporary build must remain in its cache.")
        for source in sorted(report.rglob("*")):
            include = source.suffix in {".tex", ".bib", ".png"} or (
                source.suffix == ".pdf"
                and source.parent == report / "supporting"
            )
            if not source.is_file() or not include:
                continue
            target = build / source.relative_to(report)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        subprocess.run(
            [
                str(compiler),
                "-Z",
                "deterministic-mode",
                "--keep-logs",
                "--keep-intermediates",
                "main.tex",
            ],
            cwd=build,
            env=environment,
            check=True,
        )
        for suffix in [".pdf", ".aux", ".bbl", ".blg", ".out", ".log"]:
            source = build / ("main" + suffix)
            if source.is_file():
                shutil.copy2(source, report / source.name)
    return {
        "compiler": subprocess.check_output(
            [str(compiler), "--version"], text=True
        ).strip(),
        "compiled_pdf_sha256": hashlib.sha256(
            (report / "main.pdf").read_bytes()
        ).hexdigest(),
        "build_policy": "new temporary source-only TeX workspace each time",
    }


def main() -> None:
    """Read an installed compiler path and build the existing report inputs.

    Returns:
        None. This command compiles only; it does not rerun experiments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    outcome = build_pdf(
        args.tectonic.resolve(), args.report.resolve(), ROOT / ".build-cache"
    )
    print(outcome)


if __name__ == "__main__":
    main()
